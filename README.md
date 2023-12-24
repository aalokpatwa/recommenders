# Recommender Systems
### A project for ESE 5460: Principles of Deep Learning

This repository contains from-scratch PyTorch implementations for two different deep learning architectures for recommendation.
There is also provided code for training and evaluating both models on the MovieLens-1M dataset.

To run, create a new directory at the root called data, containing files: `ml_1m_ratings.txt`, `ml_1m_users.txt`, `ml_1m_movies.txt`,
which are the original MovieLens files for movie ratings, user features, and movie features.

Then, navigate into each model's diretory and run: 
`python3 train.py`
