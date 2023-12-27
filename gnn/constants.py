# Data constants
DATASET = "movielens" # either 'movielens' or 'amzn_fashion'
MIN_REVIEWS_PER_USER = 6
RATING_THRESHOLD = 5
JSONPATH = "../data/AMAZON_FASHION.json"
MLPATH = "../data/ml_1m_ratings.txt"

# Model constants
EMBEDDING_DIM = 32
N_LAYERS = 2
OUTPUT_EMBEDDING = "mean"

# Training constants
EPOCHS = 12
BATCH_SIZE = 32
DECAY = 0.00001
LR = 0.01
K = 5

FIG_PATH = "./figures/"

# Define a random seed to use throughout
RANDOM_SEED = 42