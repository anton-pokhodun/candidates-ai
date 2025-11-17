"""Configuration settings for the candidate search agent."""

# ChromaDB settings
COLLECTION_NAME = "csv"
CHROMA_DB_PATH = "./chroma_db"

# LLM settings
LLM_MODEL = "gpt-4o-mini"

# Document processing settings
DATA_DIR = "./data"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"

METADATA_FILE = "data/candidate_metadata.json"

# =====================================================================
# ANONYMIZED NAMES
# =====================================================================
FAMOUS_NAMES = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "Isaac Newton",
    "Nikola Tesla",
    "Ada Lovelace",
    "Alan Turing",
    "Grace Hopper",
    "Stephen Hawking",
    "Carl Sagan",
    "Neil Armstrong",
    "Sally Ride",
    "Rosa Parks",
    "Martin Luther King Jr.",
    "Nelson Mandela",
    "Mahatma Gandhi",
    "Mother Teresa",
    "Malala Yousafzai",
    "Winston Churchill",
    "Abraham Lincoln",
    "George Washington",
    "Thomas Edison",
    "Alexander Graham Bell",
    "Wright Brothers",
    "Henry Ford",
    "Steve Jobs",
    "Bill Gates",
    "Elon Musk",
    "Mark Zuckerberg",
    "Jeff Bezos",
    "Oprah Winfrey",
    "Walt Disney",
    "Pablo Picasso",
    "Vincent van Gogh",
    "Frida Kahlo",
    "Claude Monet",
    "Wolfgang Mozart",
    "Ludwig van Beethoven",
    "Johann Bach",
    "Elvis Presley",
    "Michael Jackson",
    "The Beatles",
    "Bob Dylan",
    "Freddie Mercury",
    "Muhammad Ali",
    "Serena Williams",
    "Lionel Messi",
    "Michael Jordan",
    "Bruce Lee",
    "Jane Austen",
    "William Shakespeare",
    "Charles Dickens",
    "Mark Twain",
    "Ernest Hemingway",
    "Maya Angelou",
    "J.K. Rowling",
    "Charles Darwin",
    "Galileo Galilei",
    "Copernicus",
    "Johannes Kepler",
    "Benjamin Franklin",
    "Eleanor Roosevelt",
    "Cleopatra",
    "Julius Caesar",
    "Alexander the Great",
    "Napoleon Bonaparte",
    "Queen Elizabeth I",
    "Catherine the Great",
    "Confucius",
    "Buddha",
    "Socrates",
    "Plato",
    "Aristotle",
    "Pythagoras",
]
