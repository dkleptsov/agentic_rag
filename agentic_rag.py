""" An example of a Agentic RAG-based QA system using Langchain. """
import os
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain import hub
from langchain.tools.retriever import create_retriever_tool

from langchain_chroma import Chroma

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import (
    DATABASE_DIR,
    DATA_FILE,
    GRADER_MODEL,
    REWRITE_MODEL,
    GENERATE_MODEL,
    AGENT_MODEL,
    RAG_PROMPT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    QUESTION_ANSWER_PAIRS,
)

load_dotenv()

class AgentState(TypedDict):
    """
    The state of the agent.
    Attributes:
        messages (Sequence[BaseMessage]): A sequence of messages representing 
        the conversation history.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

class Grade(BaseModel):
    """A Pydantic model representing a binary relevance score.
    This model is used to validate and structure the output of the relevance grading process.
    It contains a single field, `binary_score`, which indicates whether a document is relevant
    to the user's question.
    Attributes:
        binary_score (str): A binary score indicating relevance. Must be either "yes" or "no".
                            - "yes": The document is relevant to the question.
                            - "no": The document is not relevant to the question.
    """
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

def get_vectorstore():
    """
    Get or create a vectorstore for document retrieval.
    If the vectorstore already exists, it loads it from the specified directory.
    Otherwise, it creates a new vectorstore by reading and processing the text file.
    Returns:
        Chroma: A Chroma vectorstore instance.
    """
    if os.path.exists(DATABASE_DIR):
        vectorstore = Chroma(
            persist_directory=DATABASE_DIR,
            embedding_function=OpenAIEmbeddings()
        )
    else:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            text = file.read()

        docs_list = [Document(page_content=text)]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        doc_splits = text_splitter.split_documents(docs_list)

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            # collection_name="rag-chroma",
            embedding=OpenAIEmbeddings(),
            persist_directory=DATABASE_DIR
        )
    return vectorstore

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.
    Args:
        state (messages): The current state
    Returns:
        str: A decision for whether the documents are relevant or not
    """
    model = ChatOpenAI(temperature=0, model=GRADER_MODEL, streaming=True)
    llm_with_tool = model.with_structured_output(Grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document\
        to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the\
        user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document\
        is relevant to the question.""",
        input_variables=["context", "question"],
    )
    # Chain
    chain = prompt | llm_with_tool
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score
    if score == "yes":
        return "generate"
    else:
        return "rewrite"

def rewrite(state):
    """
    Rewrite the user's question to improve clarity and relevance.
    Args:
        state (messages): The current state
    Returns:
        dict: The updated state with re-phrased question
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
    content=f""" \n
    Look at the input and try to reason about the underlying semantic intent/meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]
    model = ChatOpenAI(temperature=0, model=REWRITE_MODEL, streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    """
    Generate an answer based on the retrieved documents and user's question.
    Args:
        state (AgentState): The current state of the agent, containing the conversation history.
    Returns:
        dict: The updated state with the generated response.
    """
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    prompt = hub.pull(RAG_PROMPT)
    # print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
    # prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like
    llm = ChatOpenAI(model_name=GENERATE_MODEL, temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.
    Args:
        state (messages): The current state
    Returns:
        dict: The updated state with the agent response appended to messages
    """
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model=AGENT_MODEL)
    # model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

def ask_question(question: str) -> str:
    """Ask a question and get an answer using the RAG model.
    Args:
        question (str): The user's question.
    Returns:
        str: The answer generated by the RAG model.
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "anna_karenina",
        "Search and return information",
    )
    # tools = [retriever_tool]
    workflow = StateGraph(AgentState)        # Define a new graph
    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)        # agent
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)    # Re-writing the question
    workflow.add_node("generate", generate)  # Generating a response
    workflow.add_edge(START, "agent")        # Call agent node to decide to retrieve or not
    workflow.add_conditional_edges("agent", tools_condition,{"tools": "retrieve", END: END,})
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    # Compile
    graph = workflow.compile()
    inputs = {"messages": [("user", question),]}
    state = graph.invoke(inputs, {"recursion_limit": 10})
    return state["messages"][-1].content

if __name__ == "__main__":
    for test_question, true_answer in QUESTION_ANSWER_PAIRS:
        print(f"Question: {test_question}")
        rag_answer = ask_question(test_question)
        print("RAG answer:", rag_answer)
        print(f"Correct answer: {true_answer}")
        print("*" * 20, "\n")
