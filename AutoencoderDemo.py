import numpy as np
import math
from scipy.stats import entropy as KL
from util import *
from os.path import join

np.set_printoptions(suppress=True)

NUM_TRAIN = 320
NUM_VALID = 32
NUM_TEST = 32
BATCH_SIZE = 32
EPOCH_SIZE = NUM_TRAIN / BATCH_SIZE
NUM_ITER = EPOCH_SIZE * 250 
NUM_LATENT1 = 300
NUM_LATENT2 = 300
DECAY_RATE = 10
DATA_DIR = '/home/willa/movie'

TRAIN_PROB = 0.8 
NOISE_TRAIN_PROB = 0.0
VALID_PROB = 0.1
TEST_PROB = 0.1

alpha = 0.01 # learning rate
beta = 0.002 # penalization
l1l2 = 0.9
gamma = 0.01 # wrong value
std = 0.5

alpha_W = 0.9
beta_W = 0.99
alpha_b = 0.9
beta_b = 0.99

use_saved_model = False 

class AutoEncoder(object):
  def __init__(self):
    self.ratings, self.train_ratings, self.noise_train_ratings, self.validate_ratings, self.test_ratings = \
       get_ratings(join(DATA_DIR, 'ratings.csv'), TRAIN_PROB, NOISE_TRAIN_PROB, VALID_PROB, TEST_PROB)
    self.train_position = 0
    self.valid_position = NUM_TRAIN 
    self.test_position = NUM_TRAIN + NUM_VALID 
    num_users = np.shape(self.ratings)[0]
    self.params = {}
    self.params['W1'] = std * np.random.randn(num_users, NUM_LATENT1)
    self.params['b1'] = np.zeros(NUM_LATENT1)
    #self.params['W2'] = std * np.random.randn(NUM_LATENT1, NUM_LATENT2) 
    #self.params['b2'] = np.zeros(NUM_LATENT2)
    self.params['W3'] = std * np.random.randn(NUM_LATENT1, num_users)
    self.params['b3'] = np.zeros(num_users)
  
  def loss(self, X, reg = 0.001, sparsity = 0.1):
    W1, b1 = self.params['W1'], self.params['b1']
    #W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    N, D = X.shape
  
    # forward					# N * I (I: input length)
    Z1 = X.dot(W1) + b1
    H1 = sigmoid(Z1)  		# N * H1
    #Z2 = H1.dot(W2) + b2
    #H2 = sigmoid(Z2) 				# N * H2
    #Z3 = H2.dot(W3) + b3
    Z3 = H1.dot(W3) + b3
    O = np.maximum(0, Z3)        		# N * I (== O) 
   
    mask = (X > 0) * 1
    diff = (X - O) * mask
    roundO = np.round(O * 2) / 2
    bigO = (roundO > 5) * 1
    smallO = (roundO <= 0) * 1
    wrongO = (bigO + smallO) * O * mask
    loss = 0.5 * np.sum(np.square(diff)) / N + \
	   0.5 * beta * (1 - l1l2) * (np.mean(W1 * W1) + np.mean(W3 * W3)) / 3 + \
	   0.5 * beta * l1l2* (np.mean(np.absolute(W1)) + np.mean(np.absolute(W3))) / 3 + \
	   0.5 * gamma * np.mean(wrongO * wrongO)
  
    # backward
    grads = {}
    dO = -diff.astype(float) * mask / N
    dO += gamma * wrongO * (bigO + smallO) * mask / np.size(wrongO)
    dO[Z3 <= 0] = 0 # dZ3
    dZ3 = dO
    absW3 = W3.copy()
    absW3[W3 > 0] = 1
    absW3[W3 < 0] = -1
    grads['W3'] = clip_norm(H1.T.dot(dZ3) / N + beta * ((1 - l1l2) * W3 + l1l2 * absW3) / 3 / np.size(W3), 0) 
    #grads['W3'] = clip_norm(H2.T.dot(dZ3) / N + beta * ((1 - l1l2) * W3 + l1l2 * absW3) / 3 / np.size(W3), 0) 
    grads['b3'] = np.sum(dZ3, axis = 0)
 	
    ''' 
    dH2 = dZ3.dot(W3.T)# + drho_h2
    dZ2 = dH2 * sigmoid_prime(Z2) 
    absW2 = W2.copy()
    absW2[W2 > 0] = 1
    absW2[W2 < 0] = -1
    grads['W2'] = clip_norm(H1.T.dot(dZ2) / N + beta * ((1 - l1l2) * W2 + l1l2 * absW2) / 3 / np.size(W2), 0)
    grads['b2'] = np.sum(dZ2, axis = 0)
    '''
 
    dH1 = dZ3.dot(W3.T)
    dZ1 = dH1 * sigmoid_prime(Z1)
    absW1 = W1.copy()
    absW1[W1 > 0] = 1
    absW1[W1 < 0] = -1
    grads['W1'] = clip_norm(X.T.dot(dZ1) / N + beta * ((1 - l1l2) * W1 + l1l2 * absW1) / 3 / np.size(W1), 0)
    grads['b1'] = np.sum(dZ1, axis = 0)
   
    return loss, grads
   
  def train_and_test(self, update = 'Adam', learning_rate = 0.01, reg_rate = 0.001, decay = 0.9):
    loss_hist = []
    train_acc = []
   
    m_W1 = np.zeros_like(self.params['W1'])
    #m_W2 = np.zeros_like(self.params['W2'])
    m_W3 = np.zeros_like(self.params['W3'])
    m_b1 = np.zeros_like(self.params['b1'])
    #m_b2 = np.zeros_like(self.params['b2'])
    m_b3 = np.zeros_like(self.params['b3'])
    n_W1 = np.zeros_like(self.params['W1'])
    #n_W2 = np.zeros_like(self.params['W2'])
    n_W3 = np.zeros_like(self.params['W3'])
    n_b1 = np.zeros_like(self.params['b1'])
    #n_b2 = np.zeros_like(self.params['b2'])
    n_b3 = np.zeros_like(self.params['b3'])
    
    for i in range(NUM_ITER):
      X_batch = self.get_train_batch()
      lossval, grads = self.loss(X_batch, reg_rate)
      if update == 'SGD':
        self.params = update_params_sgd(self.params, grads, learning_rate)
      else:
        self.params = update_params_adam(self.params, grads, learning_rate, 
                     			 m_W1, m_W3, m_b1, m_b3,
                     			 n_W1, n_W3, n_b1, n_b3, 
       	    				 alpha_W, beta_W, alpha_b, beta_b)
      O,acc = self.evaluate(X_batch) 
      train_acc.append(acc)
      loss_hist.append(lossval)
      if i % EPOCH_SIZE == EPOCH_SIZE - 1:
        X_batch_valid = self.get_validate_batch()
    	O,validate_acc = self.evaluate(X_batch_valid) 
 	print 'Epoch ' + str((i + 1) / EPOCH_SIZE) 
    	print 'Training accuracy: ' + str(np.round(np.mean(train_acc),4)) + \
              ' || Training loss: ' + str(np.round(np.mean(loss_hist),4)) + \
              ' || Validation accuracy: ' + str(np.round(validate_acc,4)) 
        if i % (DECAY_RATE * EPOCH_SIZE) == (DECAY_RATE * EPOCH_SIZE - 1):
          learning_rate *= decay
          reg_rate *= decay
        loss_hist = []
        train_acc = []
 
  def evaluate(self, X):
    H1 = sigmoid(X.dot(self.params['W1']) + self.params['b1'])
    #H2 = sigmoid(H1.dot(self.params['W2']) + self.params['b2'])
    #O = np.maximum(0, H2.dot(self.params['W3']) + self.params['b3'])
    O = np.maximum(0, H1.dot(self.params['W3']) + self.params['b3'])
    O = np.round(O * 2) / 2
    mask = (X > 0) * 1
    #acc = float(np.sum(((O * mask) - y) ** 2)) / np.sum(mask)
    acc = float(np.sum((X == O) * 1 * mask)) / np.sum(mask) 
    return O, acc 
    
  def predict(self):
    X_batch_test = self.get_test_batch()
    pred, acc = self.evaluate(X_batch_test)
    pred = np.round(pred, 1)
    print pred[0,]
    print X_batch_test[0,]
    print 'Test accuracy: ' + str(acc)
    #np.savetxt('./Data/pred.txt', pred, fmt = '%4.1f', delimiter = ',') 
    #np.savetxt('./Data/true.txt', X_batch_test, fmt = '%4.1f', delimiter = ',') 

  # for autoencoder, X and y is the same
  def get_train_batch(self):
    if self.train_position == NUM_TRAIN:
      self.train_position = 0
    batch = self.ratings[:,self.train_position:self.train_position+BATCH_SIZE]
    self.train_position += BATCH_SIZE
    return batch.T

  def get_validate_batch(self):
    if self.valid_position == NUM_TRAIN + NUM_VALID:
      self.valid_position = NUM_TRAIN
    batch = self.ratings[:,self.valid_position:self.valid_position+NUM_VALID]
    return batch.T

  def get_test_batch(self):
    if self.test_position == NUM_TRAIN + NUM_VALID + NUM_TEST:
      self.test_position = NUM_TRAIN + NUM_VALID
    batch = self.ratings[:,self.test_position:self.test_position+NUM_TEST]
    return batch.T

if __name__ == '__main__':
  ae = AutoEncoder()
  ae.train_and_test(learning_rate = alpha, reg_rate = beta)
  ae.predict()
