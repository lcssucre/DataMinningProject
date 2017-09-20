import numpy as np
from util import *
from os.path import join

np.set_printoptions(suppress=True)

NUM_ITER = 100000
alpha = 1e-6 # learning rate
beta = 1e-7 # regularization 

TRAIN_PROB = 0.9 
NOISE_TRAIN_PROB = 0.0
VALID_PROB = 0.1
TEST_PROB = 0.0

DATA_DIR = '/home/willa/movie'

'''
Each row represents one user, each column represents one movie.
'''
allrates, train, noise_train, validate, test = get_ratings(join(DATA_DIR, 'sample.csv'), TRAIN_PROB, NOISE_TRAIN_PROB, VALID_PROB, TEST_PROB)

num_users = np.shape(allrates)[0]
num_movies = np.shape(allrates)[1]
# randomly initialize P and Q, code start
P = np.random.rand(num_users,300)
Q = np.random.rand(300,num_movies)
# load P and Q from last run
#P = np.loadtxt(join(DATA_DIR, 'P'), delimiter = ',')
#Q = np.loadtxt(join(DATA_DIR, 'Q'), delimiter = ',')

# mask out unobserved ratings
all_mask = (allrates > 0) * 1
train_mask = (train > 0) * 1
validate_mask = (validate > 0) * 1
test_mask = (test > 0) * 1

num_rates = np.sum(all_mask)
num_train = np.sum(train_mask)
num_validate = np.sum(validate_mask)
num_test = np.sum(test_mask)

print num_rates 
print num_train
print num_validate
print num_test

'''
Training (and validate)
'''
train_acc_hist = [1.0]
validate_acc_hist = [0.0]
for i in range(NUM_ITER):
  pred = np.dot(P, Q)
  diff = (train - pred) * train_mask
  err = 0.5 * np.sum(np.square(diff)) + beta * 0.5 * (np.sum(np.square(P)) + np.sum(np.square(Q)))
  ddiff = -diff * train_mask
  P -= alpha * (ddiff.dot(Q.T) + beta * P)
  Q -= alpha * (P.T.dot(ddiff) + beta * Q)
  if i % 10 == 0:
    pred = np.round(pred * 2) / 2
    train_acc = float(np.sum((pred == train) * 1 * train_mask)) / num_train
    # save P and Q 
    #if train_acc > train_acc_hist[-1]:
    #  np.savetxt(join(DATA_DIR, 'P'), P, fmt = '%10.8f', delimiter = ',') 
    #  np.savetxt(join(DATA_DIR, 'Q'), Q, fmt = '%10.8f', delimiter = ',')
    train_acc_hist.append(train_acc) 
    validate_acc = float(np.sum((pred == validate) * 1 * validate_mask)) / num_validate
    validate_acc_hist.append(validate_acc) 
    print '###########################################'
    print 'Iteration ' + str(i)
    print 'err: ' + str(err)
    print 'Train accuracy: ' + str(train_acc)
    print 'Validate accuracy: ' + str(validate_acc)
    print '###########################################'

'''
Test
'''
P = np.loadtxt(join(DATA_DIR, 'P'), delimiter = ',')
Q = np.loadtxt(join(DATA_DIR, 'Q'), delimiter = ',')
pred = np.dot(P, Q)
pred = np.round(pred * 2) / 2
test_acc = float(np.sum((pred * test_mask == test) * 1 * test_mask)) / num_test
print '###########################################'
print 'Test accuracy: ' + str(test_acc)
print '###########################################'
