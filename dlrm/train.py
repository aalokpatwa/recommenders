import pandas as pd
import numpy as np

import torch
from torch.nn import functional as F

from matplotlib import pyplot as plt

from constants import *
from data import *
from dlrm import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

experiment_string = f"dataset_{DATASET}__embed{EMBEDDING_DIM}_decay{DECAY}_tfidf{MAX_TFIDF_FEATURES}"

print (f"On device: {device}")

def get_test_loss(model, test_df):
    """ Obtains the MSE loss on the test dataset"""
    losses = []
    model.eval()
    with torch.no_grad():
        num_samples = len(test_df)
        batches = num_samples // BATCH_SIZE

        for batch in range(batches):
            # Fetch the features for the test dataframe
            continuous, categorical, labels = data_loader(test_df, BATCH_SIZE)
            logits = model(continuous, categorical)

            loss = F.mse_loss(logits, labels.unsqueeze(-1))
            losses.append(loss.item())

    model.train()
    return np.mean(np.array(losses))

def get_hit_p(model, data, product_features, user_features):
    """ Helper function to perform recommendations and calculate hit percentage

    Args:
        model (nn.Module): DLRM model
        data (pd.DataFrame): train or test dataframe containing ratings
        product_features (pd.DataFrame): dataframe containing features for each movie
        user_features (pd.DataFrame): dataframe containing features for each user

    Returns:
        float: the hit percentage for the users in this dataset
    """

    # From the raw ratings data, collect all of the 5/5 ratings each user has given
    liked_movies = data[data['overall'] == 5]
    liked_per_user = liked_movies.groupby("userID")["movieID"].apply(list)

    user_preds = []

    num_products = len(product_features.index)

    # Loop through each unique user
    for user in data["userID"].unique():

        this_user_features = np.expand_dims(user_features.loc[user, :].to_numpy(), axis=0)
       
        this_user_scores = torch.zeros(product_features.index.max() + 1)
        
        for product in range(0, num_products, BATCH_SIZE):
            # We will predict product rankings in batches
            product_indices = range(product, min(product + BATCH_SIZE, num_products - 1))

            movies = product_features.index[product_indices]

            # Concatenate each product's features with the user's feeatures
            repeated = np.repeat(this_user_features, len(product_indices), axis=0)
            this_product_features = product_features.iloc[product_indices, :]
            
            genres = np.expand_dims(this_product_features["genres"].to_numpy(), axis=1)
            
            categorical = np.concatenate([repeated, genres], axis=1)
            
            continuous = this_product_features.drop(columns=["genres", "overall", "index"]).to_numpy()

            # Separate into continuous and categorical features
            continuous = torch.FloatTensor(continuous).to(device)
            categorical = torch.LongTensor(categorical).to(device)

            scores = model(continuous, categorical)

            # Update the predicted scores of these products
            this_user_scores[movies] = scores.squeeze().cpu()
        
        # Get the top-5 rater movies for this user
        top5 = torch.topk(this_user_scores, K,  dim=0).indices.tolist()
        user_preds.append([user, top5])

    predicted = pd.DataFrame(user_preds, columns=["userID", "predicted"])

    # Count the overlap between each user's liked movies and our predictions
    merged = pd.merge(liked_per_user, predicted, how="inner", on="userID")
    merged['intersect'] = [list(set(a).intersection(b)) for a, b in zip(merged.movieID, merged.predicted)]
    has_intersection = (merged['intersect'].apply(len) > 0).sum()

    hit_p = has_intersection / len(merged)

    return hit_p


def train(model, optimizer, train_df, test_df):
    """ Main training loop

    Returns:
        tuple: train/test loss, train/test hit percentage over training
    """
    train_loss = []
    train_hit_p_list = []

    test_loss = []
    test_hit_p_list = []
    
    for epoch in range(EPOCHS):

        # Step-function learning rate schedule
        if epoch == EPOCHS // 2:
            print (f"Changing learning rate to {LR / 5}")
            for g in optimizer.param_groups:
                g['lr'] = LR / 5
        if epoch == EPOCHS // 4 * 3:
            print (f"Changing learning rate to {LR / 25}")
            for g in optimizer.param_groups:
                g['lr'] = LR / 25
        if epoch == EPOCHS // 8 * 7:
            print (f"Changing learning rate to {LR / 25}")
            for g in optimizer.param_groups:
                g['lr'] = LR / 50

        # Number of steps per epoch
        n_batch = int(len(train_df)/BATCH_SIZE)
    
        train_loss_list = []
        test_loss_list = []
    
        model.train()
        for batch_idx in range(n_batch):
            optimizer.zero_grad()

            # Load indices of users, positive items, negative items in the batch
            continuous, categorical, labels = data_loader(train_df, BATCH_SIZE)
            continuous = continuous.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device)

            logits = model(continuous, categorical)
            

            loss = F.mse_loss(logits, labels.unsqueeze(-1))
            #loss = F.cross_entropy(logits, labels)

            # Update weights
            loss.backward()
            optimizer.step()

            # Record train loss in the middle of the epoch
            if (batch_idx == n_batch // 2):
                train_loss_list.append(loss.item())                
                
        # End of each epoch: get metrics on the validation set
        model.eval()
        with torch.no_grad():
            test_loss_amount = get_test_loss(model, test_df)
            test_loss_list.append(test_loss_amount)
            print (f"Epoch: {epoch} Step: {batch_idx}/{n_batch}. Train Loss: {np.mean(np.array(train_loss_list)[-300:])}. Test Loss: {test_loss_amount}")
            

        train_loss.append(round(np.mean(train_loss_list), 4))
        test_loss.append(round(np.mean(test_loss_list), 4))

    train_hit_p = get_hit_p(model, train_df, product_features, user_features)
    test_hit_p = get_hit_p(model, test_df, product_features, user_features)
    print (f"Train Hit P: {train_hit_p}. Test Hit P: {test_hit_p}")
    train_hit_p_list.append(train_hit_p)
    test_hit_p_list.append(test_hit_p)
    
    return (
        train_loss,
        train_hit_p_list,
        test_loss,
        test_hit_p_list
    )

# Read and parse the data
all_ratings = read_movielens(RATINGSPATH, MOVIESPATH, USERSPATH)

train_df, test_df, product_features, user_features, n_continuous, n_categorical, levels = get_train_test(all_ratings)

dlrm = DLRM(n_continuous, n_categorical, levels)

dlrm = dlrm.to(device)

trainable_parameters = filter(lambda p: p.requires_grad, dlrm.parameters())
params = sum([np.prod(p.size()) for p in trainable_parameters])

print (f"\nModel has {params} parameters\n")

optimizer = torch.optim.Adam(dlrm.parameters(), lr=LR, weight_decay=DECAY)

train_loss, train_hit_p, test_loss, test_hit_p = train(dlrm, optimizer, train_df, test_df)

plt.plot(range(EPOCHS), train_loss, label='Training MSE Loss', color="blue")
plt.plot(range(EPOCHS), test_loss, label='Test MSE Loss', color="orange")

plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend()

plt.savefig(FIG_PATH + "dlrm_loss_" + experiment_string + ".png")

plt.clf()

plt.plot(range(len(train_hit_p)), train_hit_p, label='Train Hit Percentage')
plt.plot(range(len(test_hit_p)), test_hit_p, label='Test Hit Percentage')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()

plt.savefig(FIG_PATH + "dlrm_hitpercentage_" + experiment_string + ".png")
