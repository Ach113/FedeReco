import os

from spec import *
from train import train_model
from dataset import Dataset
from eval import evaluate_model
from models import default_model


def main():
    # import the dataset
    dataset = Dataset(DATA_PATH)
    print('Reading MovieLens 1M dataset')
    train, test_ratings, test_negatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    # define model
    model = default_model(num_users, num_items)
    print('Building the model')
    if os.path.exists(SAVED_MODEL_PATH):
        model.load_weights(SAVED_MODEL_PATH)
    else:
        model = train_model(train)
    # evaluate the model
    print('Evaluating the model')
    hr, ndcg = evaluate_model(model, test_ratings, test_negatives, k=10, n_threads=1)
    print(f'Hit Rate: {round(hr, 4)}\nNormalized Discounted Cumulative Gain: {round(ndcg, 4)}')


if __name__ == '__main__':
    main()
