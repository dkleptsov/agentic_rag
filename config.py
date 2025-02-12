""" Configuration file for the Agentic RAG project. """
# Database
DATABASE_DIR = "chroma_db"
DATA_FILE = "data/anna-karenina.txt"
COLLECTION_NAME = "anna-karenina"

# Text splitting settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Models and parameters
GRADER_MODEL = "gpt-4o"
REWRITE_MODEL="gpt-4o-2024-08-06"
GENERATE_MODEL = "gpt-4o-mini"
AGENT_MODEL = "gpt-4-turbo"
OPENAI_TEMPERATURE = 0

# RAG prompt
RAG_PROMPT = "rlm/rag-prompt"

QUESTION_ANSWER_PAIRS = [
    # ("What is Anna's full name?", "Anna Arkadyevna Karenina"),
    ("Who is Stepan Oblonsky's wife?", "Darya Alexandrovna (Dolly)"),
    # ("What is the name of Anna's son?", "Sergei (Seryozha) Karenin"),
    # ("Who is Konstantin Levin in love with?", "Kitty (Ekaterina Shcherbatskaya)"),
    # ("What happens to Anna at the end of the novel?", "She commits suicide by throwing herself under a train."),
    # ("What is the name of Vronsky's horse that dies during a race?", "Frou-Frou"),
    # ("Who proposes to Kitty twice before she accepts?", "Konstantin Levin"),
    # ("Who convinces Karenin to forgive Anna at one point?", "Countess Lydia Ivanovna"),
    # ("What philosophical topic does Levin struggle with throughout the novel?", "The meaning of life"),
    # ("What does Levin do for a living?", "He is a landowner and farmer."),
    # ("What substance does Anna start using heavily near the end of the novel?", "Opium"),
    # ("What is the fate of Vronsky after Anna's death?", "He joins the Russian army and goes to fight in Serbia."),
    # ("What is the name of Levin's half-brother who is a writer?", "Sergei Ivanovich Koznyshev"),
    # ("What game does Levin play with Kitty when they are falling in love?", "A word game with chalk on a table"),
    # ("What does Dolly discover about her husband Oblonsky?", "He is having an affair with their children’s governess."),
    # ("Which city does Anna visit with Vronsky to escape society’s judgment?", "Rome"),
    # ("What is Karenin's profession?", "A high-ranking government official"),
    # ("Who accompanies Anna Karenina on her fateful final train ride?", "No one, she travels alone."),
    # ("What activity does Levin enjoy that connects him with the peasants?", "Mowing the fields"),
    # ("Who acts as a mentor figure for Levin?", "His half-brother Sergei Ivanovich Koznyshev"),
    # ("Who is Kitty’s father?", "Prince Shcherbatsky"),
    # ("What reason does Anna Karenina give for her unhappiness before her death?", "Vronsky's growing coldness toward her"),
    # ("What significant event happens at a train station early in the novel?", "A railway worker is accidentally killed."),
    # ("How does Anna Karenina feel after leaving her husband for Vronsky?", "Guilt and social isolation"),
    # ("What does Levin struggle with before marrying Kitty?", "Self-doubt and existential crisis"),
    # ("What is the name of Anna Karenina daughter with Vronsky?", "Annie"),
    # ("Who helps take care of Anna  Karenina son after she leaves Karenin?", "Karenin himself"),
    ]
