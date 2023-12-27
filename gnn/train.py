import pandas as pd
import numpy as np

import torch
from torch.nn import functional as F

from matplotlib import pyplot as plt

from constants import *
from lightgcn import RecSysGNN
from data import *

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

experiment_string = f"dataset_{DATASET}_thresh{RATING_THRESHOLD}_embed{EMBEDDING_DIM}_decay{DECAY}_layers{N_LAYERS}"

print (f"Starting training run with settings: \n min reviews per user: \
       {MIN_REVIEWS_PER_USER} \n rating threshold: {RATING_THRESHOLD} \n embedding dim: {EMBEDDING_DIM} \
       \n weight decay: {DECAY} \n n_layers {N_LAYERS}")

print (f"On device: {device}")

def compute_bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0,  pos_emb0, neg_emb0):
    """ Computes BPR and regularization loss for the generated embeddings """
    # compute L2 norm of the initial parameters
    # This is akin to weight decay since the starting embeddings are the parameters
    reg_loss = (user_emb0.norm().pow(2) + pos_emb0.norm().pow(2)  + neg_emb0.norm().pow(2)) / (2 * float(len(users)))

    # compute positive/negative item scores for users in the batch
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)

    # Take the average of the difference in neg and pos scores across the users in the batch
    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
    return bpr_loss, reg_loss

def get_train_metrics(user_embeds, item_embeds, train_df, K):
    # compute the "scores" between all users and items
    # Relevance score is of shape (n_users * n_items)
    relevance_score = torch.matmul(user_embeds, torch.transpose(item_embeds,0, 1))

    # compute top K scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(),columns =['top_indx_'+str(x+1) for x in range(K)])
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID','top_rlvnt_itm']]

    # Get the list of truly interacted-with items per user in the test set
    interacted_items = train_df.groupby('user_idx')['item_idx'].apply(set).reset_index()

    # Join this df with the top-k recommended items per user
    metrics_df = pd.merge(interacted_items, topk_relevance_indices_df, how= 'left', left_on = 'user_idx', right_on = ['user_ID'])

    # Find the intersection of top-k recommended items and true positive interactions
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_idx, metrics_df.top_rlvnt_itm)]

    total_intersection = metrics_df['intrsctn_itm'].apply(len).sum()
    has_intersection = (metrics_df['intrsctn_itm'].apply(len) > 0).sum()
    total_liked = metrics_df['item_idx'].apply(len).sum()

    recall = total_intersection / total_liked
    precision = total_intersection / (K * len(metrics_df))
    has_hit = has_intersection / len(metrics_df)
    
    # Return all four metrics: mean recall, mean precision, average rank of liked items, and % of times that a liked item was in the set
    return recall, precision, has_hit

def get_test_metrics(user_embeds, item_embeds, n_users, n_items, train_df, test_df, K):
  
    # compute the "scores" between all users and items
    # Relevance score is of shape (n_users * n_items)
    relevance_score = torch.matmul(user_embeds, torch.transpose(item_embeds,0, 1))

    # create dense tensor of all user-item interactions
    i = torch.stack((
        torch.LongTensor(train_df['user_idx'].values),
        torch.LongTensor(train_df['item_idx'].values)
    ))
    v = torch.ones((len(train_df)), dtype=torch.float64)
    interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items)).to_dense().to(device)

    # mask out training user-item interactions from metric computation
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))

    # compute top K scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(),columns =['top_indx_'+str(x+1) for x in range(K)])
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID','top_rlvnt_itm']]

    # Get the list of truly interacted-with items per user in the test set
    test_interacted_items = test_df.groupby('user_idx')['item_idx'].apply(set).reset_index()

    # Join this df with the top-k recommended items per user
    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how= 'left', left_on = 'user_idx', right_on = ['user_ID'])

    # Find the intersection of top-k recommended items and true positive interactions
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_idx, metrics_df.top_rlvnt_itm)]

    total_intersection = metrics_df['intrsctn_itm'].apply(len).sum()
    has_intersection = (metrics_df['intrsctn_itm'].apply(len) > 0).sum()
    total_liked = metrics_df['item_idx'].apply(len).sum()

    test_recall = total_intersection / total_liked
    test_precision = total_intersection / (K * len(metrics_df))
    has_hit = has_intersection / len(metrics_df)

    metrics_df.to_csv(FIG_PATH + "preds.csv")
    
    # Return all four metrics: mean recall, mean precision, average rank of liked items, and % of times that a liked item was in the set
    return test_recall, test_precision, has_hit

def get_test_loss(model, train_df, test_df, n_users, n_items, train_edge_index):
    interacted_items = train_df.groupby('user_idx')['item_idx'].apply(list).reset_index()
    test_interacted_items = test_df.groupby('user_idx')['item_idx'].apply(list).reset_index()

    all_users = test_df['user_idx'].unique()

    num_users = test_df['user_idx'].nunique()

    batches = num_users // BATCH_SIZE

    all_items = range(n_items)

    losses = []

    for batch in range(batches):
        users = [random.choice(all_users) for _ in range(BATCH_SIZE)]

        pos_items = []
        neg_items = []

        for user in users:
            test_items = test_interacted_items[test_interacted_items['user_idx'] == user]['item_idx'].values[0]
            train_items = interacted_items[interacted_items['user_idx'] == user]['item_idx'].values[0]
            both_items = test_items + train_items

            pos_item = random.choice(test_items)
            neg_item = random.choice(list(set(all_items).difference(set(both_items))))

            pos_items.append(pos_item + n_users)
            neg_items.append(neg_item + n_users)

        users = torch.LongTensor(list(users)).to(device),
        pos_items = torch.LongTensor(list(pos_items)).to(device)
        neg_items = torch.LongTensor(list(neg_items)).to(device)
        
        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = model.encode_minibatch(users, pos_items, neg_items, train_edge_index)

        bpr_loss, _ = compute_bpr_loss(
            users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0
        )

        losses.append(bpr_loss.item())

    return np.mean(np.array(losses))

def train_and_eval(model, optimizer, train_df, train_edge_index, test_df, n_users, n_items):
    loss_list_epoch = []
    bpr_loss_list_epoch = []
    reg_loss_list_epoch = []

    test_loss_list_epoch = []

    train_recall_list = []
    train_precision_list = []
    train_hit_p_list = []

    recall_list = []
    precision_list = []
    test_hit_p_list = []

    model.eval()
    
    with torch.no_grad():
        test_loss = get_test_loss(model, train_df, test_df, n_users, n_items, train_edge_index)
        print (f"\nTest loss: {test_loss}")

        _, out = model(train_edge_index)
        final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
        train_recall, train_precision, train_hit_p = get_train_metrics(final_user_Embed, final_item_Embed, train_df, K)
        test_recall,  test_precision, test_hit_p = get_test_metrics(
            final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, K
        )
        print (f"Train Recall: {train_recall}. Precision: {train_precision}. Hit percentage: {train_hit_p}")
        print (f"Recall: {test_recall}. Precision: {test_precision}. Hit percentage: {test_hit_p}\n")

    for epoch in range(EPOCHS):
        if epoch == EPOCHS // 2:
            print (f"Changing learning rate to {LR / 5}")
            for g in optimizer.param_groups:
                g['lr'] = LR / 5
        if epoch == EPOCHS // 4 * 3:
            print (f"Changing learning rate to {LR / 25}")
            for g in optimizer.param_groups:
                g['lr'] = LR / 25

        # Number of steps per epoch
        n_batch = int(len(train_df)/BATCH_SIZE)
    
        final_loss_list = []
        bpr_loss_list = []
        reg_loss_list = []

        test_loss_list = []
    
        model.train()
        for batch_idx in range(n_batch):
            optimizer.zero_grad()

            # Load indices of users, positive items, negative items in the batch
            users, pos_items, neg_items = data_loader(train_df, BATCH_SIZE, n_users, n_items, device)
            users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = model.encode_minibatch(users, pos_items, neg_items, train_edge_index)

            # Compute loss
            bpr_loss, reg_loss = compute_bpr_loss(
                users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0
            )
            reg_loss = DECAY * reg_loss
            final_loss = bpr_loss + reg_loss

            # Update weights
            final_loss.backward()
            optimizer.step()

            final_loss_list.append(final_loss.item())
            bpr_loss_list.append(bpr_loss.item())
            reg_loss_list.append(reg_loss.item())

            if (batch_idx % 300 == 0):
                model.eval()
                with torch.no_grad():
                    test_loss = get_test_loss(model, train_df, test_df, n_users, n_items, train_edge_index)
                test_loss_list.append(test_loss)
                print (f"Epoch: {epoch} Step: {batch_idx}/{n_batch}. BPR Loss: {np.mean(np.array(bpr_loss_list)[-300:])}. Total Loss: {final_loss.item()} Test Loss: {test_loss}")
                model.train()
        # End of each epoch: get metrics on the validation set
        model.eval()
        with torch.no_grad():
            _, out = model(train_edge_index)
            final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
            train_recall, train_precision, train_hit_p = get_train_metrics(final_user_Embed, final_item_Embed, n_users, n_items, train_df, K)
            test_recall,  test_precision, test_hit_p = get_test_metrics(
                final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, K
            )
            print (f"\nTrain Recall: {train_recall}. Precision: {train_precision}. Hit percentage: {train_hit_p}")
            print (f"Recall: {test_recall}. Precision: {test_precision}. Hit percentage: {test_hit_p}\n")

        loss_list_epoch.append(round(np.mean(final_loss_list),4))
        bpr_loss_list_epoch.append(round(np.mean(bpr_loss_list),4))
        reg_loss_list_epoch.append(round(np.mean(reg_loss_list),4))
        test_loss_list_epoch.append(round(np.mean(test_loss_list),4))

        train_recall_list.append(round(train_recall,4))
        train_precision_list.append(round(train_precision,4))
        train_hit_p_list.append(round(train_hit_p, 4))
        
        recall_list.append(round(test_recall,4))
        precision_list.append(round(test_precision,4))
        test_hit_p_list.append(round(test_hit_p, 4))

    return (
        loss_list_epoch, 
        bpr_loss_list_epoch, 
        reg_loss_list_epoch, 
        test_loss_list_epoch,
        train_recall_list,
        train_precision_list,
        train_hit_p_list,
        recall_list,
        precision_list,
        test_hit_p_list
    )

# Read and parse the data
if DATASET == 'movielens':
    all_ratings = read_movielens(MLPATH)
elif DATASET == 'amzn_fashion':
    all_ratings = read_amazon(JSONPATH)

positive_interactions = filter_by_ratings(all_ratings)
train_df, test_df, n_users, n_items = get_train_test(positive_interactions)

train_edge_index = get_edge_list(train_df, device)

lightgcn = RecSysGNN(
    latent_dim=EMBEDDING_DIM, 
    num_layers=N_LAYERS,
    num_users=n_users,
    num_items=n_items,
)

lightgcn.to(device)

optimizer = torch.optim.Adam(lightgcn.parameters(), lr=LR)
print("Size of Learnable Embedding : ", [x.shape for x in list(lightgcn.parameters())])

light_loss, light_bpr, light_reg, test_bpr, train_recall, train_precision, train_hit_p, test_recall, test_precision, test_hit_p \
      = train_and_eval(lightgcn, optimizer, train_df, train_edge_index, test_df, n_users, n_items)

plt.plot(range(EPOCHS), light_bpr, label='Training BPR Loss', color="blue")
plt.plot(range(EPOCHS), test_bpr, label='Test BPR Loss', color="orange")

plt.xlabel('Epoch')
plt.ylabel('BPR ')
plt.legend()

plt.savefig(FIG_PATH + "lightgcn_loss_" + experiment_string + ".png")

plt.clf()

plt.plot(range(EPOCHS), train_hit_p, label='Train Hit Percentage')
plt.plot(range(EPOCHS), test_hit_p, label='Test Hit Percentage')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()

plt.savefig(FIG_PATH + "lightgcn_hitpercentage_" + experiment_string + ".png")
