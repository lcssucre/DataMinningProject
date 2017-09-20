import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def init():
    user_to_idx = {}
    movie_to_idx = {}

    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ratings.csv', sep=',', names=names)
    # df = pd.read_csv('u.data', sep='\t', names=names)
    df.head()

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        if row[1] not in user_to_idx:
            user_to_idx[row[1]] = len(user_to_idx)
        if row[2] not in movie_to_idx:
            movie_to_idx[row[2]] = len(movie_to_idx)
        ratings[user_to_idx[row[1]], movie_to_idx[row[2]]] = row[3]

    test = np.zeros(ratings.shape)
    train = ratings.copy()

    for user in xrange(ratings.shape[0]):
        size = int(np.count_nonzero(ratings[user]) * 0.1)
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size, replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    return train, test


def fast_similarity(ratings, kind, epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


def predict_topk(ratings, similarity, kind):
    k = 20
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in xrange(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in xrange(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
            for i in xrange(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))

    return pred


def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()

    return mean_squared_error(pred, actual)


if __name__ == '__main__':
    train, test = init()
    num_test = np.sum(test)
    kind = 'item'

    user_similarity = fast_similarity(train, kind)

    pred = predict_topk(train, user_similarity, kind)
    pred = np.round(pred * 2) / 2

    mask = np.zeros(test.shape)
    mask[np.nonzero(test)] = 1
    pred *= mask

    test_acc = float(np.sum(pred) / num_test)

    print('Top-20 ' + kind + '-based CF Accuracy: ' + str(test_acc))
    print('Top-20 ' + kind + '-based CF RMSE: ' + str(get_mse(pred, test)))
