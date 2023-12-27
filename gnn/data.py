import pandas as pd
import torch
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from constants import *

def read_movielens(path: str):
    """ Reads the ratings for MovieLens from the ratings txt file """
    df = pd.read_csv(path, sep='::')
    df.columns = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
    return df

def read_amazon(path: str):
    """ Reads the ratings for Amazon Fashion from the ratings.json file"""
    # Read in data
    df = pd.read_json(path, lines=True)

    reviews_per_user = df.groupby("reviewerID")["asin"].count().sort_values(ascending=True)
    high_review_users = reviews_per_user[reviews_per_user >= MIN_REVIEWS_PER_USER].index

    # To start with, let's only use rating data.
    df = df[["overall", "reviewerID", "asin", "unixReviewTime"]]
    
    # Also filter out users and products that we don't have lots of reviews for
    five_core = df[df['reviewerID'].isin(high_review_users)]

    del df

    reviews_per_user = five_core.groupby("reviewerID")["asin"].count().sort_values(ascending=True)

    print (f"Mean reviews per user: {reviews_per_user.mean()}")

    return five_core

def filter_by_ratings(df: pd.DataFrame):
    """ Filter the entire ratings dataframe to only include positive interactions """

    print (f"Original num reviews of dataset: {len(df)}")
    
    df = df[df["overall"] >= RATING_THRESHOLD]

    print (f"Number of positive reviews in dataset: {len(df)}")

    return df

def get_train_test(df: pd.DataFrame):
    """ Split the dataset into train/test and find the number of users/items in the training set """

    # split the dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    # need to relabel the reviewers and asins to ensure the test data is a subset of the train data
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    train_df['user_idx'] = le_user.fit_transform(train_df['reviewerID'].values)
    train_df['item_idx'] = le_item.fit_transform(train_df['asin'].values)

    train_users = train_df['reviewerID'].unique()
    train_items = train_df['asin'].unique()

    # Only include users and items that are in the training set
    test_df = test_df[(test_df['asin'].isin(train_items)) & \
                    (test_df['reviewerID'].isin(train_users))
                    ]
    test_df['user_idx'] = le_user.transform(test_df['reviewerID'].values)
    test_df['item_idx'] = le_item.transform(test_df['asin'].values)

    print (f"Train size (number of edges): {len(train_df)}")
    print (f"Test size: {len(test_df)}")

    n_users = train_df['user_idx'].nunique()
    n_items = train_df['item_idx'].nunique()

    print (f"Number of unique users (nodes): {n_users}")
    print (f"Number of unique items (nodes): {n_items}")
    print ("\n")

    return train_df, test_df, n_users, n_items

def data_loader(data, batch_size: int, n_users: int, n_items: int, device: str):
    """ Returns a batch of data for LightGCN

    Args:
        data (_type_): dataframe containing ratings
        batch_size (int): batch size
        n_users (int): number of users in the graph
        n_items (int): number of items in the graph
        device (str): device to move the batch to

    Returns:
        tuple[LongTensor]: user ids, positive item ids, negative item ids in the batch
    """
    # Helper function to choose a negative (not highly rated) item
    def sample_neg(interacted):
        while True:
            neg_id = random.randint(0, n_items - 1)
            if neg_id not in interacted:
                return neg_id

    # Get the interacted items for each user
    interacted_items = data.groupby('user_idx')['item_idx'].apply(list).reset_index()
    all_users = data['user_idx'].unique()

    # Select the users in the batch
    users = [random.choice(all_users) for _ in range(batch_size)]

    users.sort()
    users_df = pd.DataFrame(users,columns = ['users'])

    # Choose the positive and negative item for each user in the batch
    interacted_items = pd.merge(interacted_items, users_df, how = 'right', left_on = 'user_idx', right_on = 'users')
    pos_items = interacted_items['item_idx'].apply(lambda x : random.choice(x)).values
    neg_items = interacted_items['item_idx'].apply(lambda x: sample_neg(x)).values

    # Returns Bx1 tensor of user ids, Bx1 tensor of positive ids, Bx1 tensor of negative ids
    return (
        torch.LongTensor(list(users)).to(device),
        torch.LongTensor(list(pos_items)).to(device) + n_users,
        torch.LongTensor(list(neg_items)).to(device) + n_users
    )

def get_edge_list(train_df: pd.DataFrame, device: str):
    """ Return a 2xM array representing the edges in the graph """
    # Define edges of the graph
    u_t = torch.LongTensor(train_df.user_idx.values)
    i_t = torch.LongTensor(train_df.item_idx.values) + train_df['user_idx'].nunique()

    train_edge_index = torch.stack((
        torch.cat([u_t, i_t]),
        torch.cat([i_t, u_t])
    )).to(device)

    return train_edge_index