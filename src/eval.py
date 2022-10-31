'''
based on code taken from: https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/evaluate.py
'''
import math
import heapq
import numpy as np
import multiprocessing

# TODO!: Seems very slow, perhaps can be optimized


def get_hit_ratio(rank_list, item):
    for x in rank_list:
        if x == item:
            return 1
    return 0


def get_ncdg(rank_list, item):
    for i, x in enumerate(rank_list):
        if x == item:
            return math.log(2) / math.log(i + 2)
    return 0


def evaluate_model(model, test_ratings, test_negatives, k, n_threads):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """

    def eval_one_rating(idx):
        rating = test_ratings[idx]
        items = test_negatives[idx]
        u = rating[0]
        item = rating[1]
        items.append(item)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u, dtype='int32')
        predictions = model.predict([users, np.array(items)], batch_size=100, verbose=0)
        for i, x in enumerate(items):
            map_item_score[x] = predictions[i]
        items.pop()

        # Evaluate top rank list
        rank_list = heapq.nlargest(k, map_item_score, key=map_item_score.get)
        hr = get_hit_ratio(rank_list, item)
        ndcg = get_ncdg(rank_list, item)
        return hr, ndcg

    hits, ndcgs = [], []
    # TODO!: fix multithreading
    if n_threads > 1:
        pool = multiprocessing.Pool(processes=n_threads)
        res = pool.map(eval_one_rating, range(len(test_ratings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return hits, ndcgs

    for i in range(len(test_ratings)):
        hr, ndcg = eval_one_rating(i)
        hits.append(hr)
        ndcgs.append(ndcg)

    return np.array(hits).mean(), np.array(ndcgs).mean()
