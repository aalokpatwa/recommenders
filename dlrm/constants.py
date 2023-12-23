# Data constants
DATASET = "movielens" # either 'movielens' or 'amzn_fashion'
MAX_TFIDF_FEATURES = 20
JSONPATH = "../data/AMAZON_FASHION.json"
RATINGSPATH = "../data/ml_1m_ratings.txt"
MOVIESPATH = "../data/ml_1m_movies.txt"
USERSPATH = "../data/ml_1m_users.txt"


# Model constants
BOTTOM_MLP_LAYERS = 15
BOTTOM_MLP_HIDDEN_DIM = 512
EMBEDDING_DIM = 512
TOP_MLP_LAYERS = 15
TOP_MLP_HIDDEN_DIM = 1024

# Training constants
EPOCHS = 30
BATCH_SIZE = 256
LR = 0.001
DECAY = 0.001


# Evaluation constants
K = 5
FIG_PATH = "./figures/"

RANDOM_SEED = 42
