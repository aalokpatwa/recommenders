import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from constants import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print (f"On device: {device}")

def read_movielens(ratings_path, movies_path, users_path):
    """ Read in ratings, user features, product features dataframes """
    ratings_df = pd.read_csv(ratings_path, sep='::', engine="python")
    ratings_df.columns = ['userID', 'movieID', 'overall', 'unixReviewTime']

    movies_df = pd.read_csv(movies_path, sep="::", encoding="latin-1", engine="python")
    movies_df.columns = ['movieID', 'title', 'genres']

    users_df = pd.read_csv(users_path, sep="::", engine="python")
    users_df.columns = ['userID', 'gender', 'age', 'occupation', 'zipcode']

    df = pd.merge(ratings_df, movies_df, how="inner", on="movieID")
    df = pd.merge(df, users_df, how="inner", on='userID')

    print (movies_df["movieID"].nunique())

    # Encode the categorical features ordinally
    df["gender"] = LabelEncoder().fit_transform(df["gender"]).astype("int")
    df["age"] = LabelEncoder().fit_transform(df["age"]).astype("int")
    df["occupation"] = LabelEncoder().fit_transform(df["occupation"]).astype("int")
    df["zipcode"] = LabelEncoder().fit_transform(df["zipcode"]).astype("int")
    df["genres"] = LabelEncoder().fit_transform(df["genres"]).astype("int")

    
    return df

def get_train_test(df: pd.DataFrame):
    # split the dataset
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    
    # Calculate tf-idf features from the movie titles
    vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES)
    train_features = vectorizer.fit_transform(train["title"].values).toarray()
    test_features = vectorizer.transform(test["title"].values).toarray()

    # Define a dataframe containing these features
    train_features_df = pd.DataFrame(train_features, columns=[f"tf-idf-{i}" for i in range(test_features.shape[1])])
    test_features_df = pd.DataFrame(test_features, columns=[f"tf-idf-{i}" for i in range(test_features.shape[1])])

    # Add to the existing train and test dataframes
    train_df = pd.concat([train.reset_index(), train_features_df], axis=1)
    test_df = pd.concat([test.reset_index(), test_features_df], axis=1)

    print (f"Number of reviews in training dataset: {len(train_df)}")

    # Count the number of levels per categorical feature (for Embedding layer)
    categorical_df = df[["gender", "age", "occupation", "zipcode", "genres"]]
    levels_per_category = []
    for column in categorical_df.columns:
        levels = categorical_df[column].nunique()
        levels_per_category.append(levels)

    # Drop unnecessary columns and create user/product features
    combined_df = pd.concat([train_df, test_df], axis=0)
    product_features = combined_df.drop_duplicates(subset="movieID", keep="last").drop(columns=["userID", "title", "gender", "age", "occupation", "zipcode", "unixReviewTime"])
    product_features.set_index("movieID", inplace=True)
    user_features = combined_df.drop_duplicates(subset="userID", keep="last")[["gender", "age", "occupation", "zipcode", "userID"]]
    user_features.set_index("userID", inplace=True)

    return train_df, test_df, product_features, user_features, MAX_TFIDF_FEATURES, len(categorical_df.columns), levels_per_category

def data_loader(data, batch_size):
    """ Fetch batches from the provided dataframe """
    indices = np.random.choice(data.index, size=(batch_size,))

    batch = data.loc[indices, :]

    continuous = batch[[f"tf-idf-{i}" for i in range(MAX_TFIDF_FEATURES)]].to_numpy().astype("float")
    categorical = batch[["gender", "age", "occupation", "zipcode", "genres"]].to_numpy().astype("int")
    labels = batch["overall"].to_numpy()

    # Change labels back to a LongTensor
    return torch.FloatTensor(continuous).to(device), torch.LongTensor(categorical).to(device), torch.FloatTensor(labels).to(device)
    
    
