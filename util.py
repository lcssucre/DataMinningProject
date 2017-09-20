import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
  return sigmoid(x) * (1 - sigmoid(x))

def KL_divergence(p, q):
  return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def SGD(O, X):
  mask = (X > 0) * 1
  diff = (O - X) * mask
  return 0.5 * np.sum(np.square(diff))

# norm must by non negative number
def clip_norm(X, norm):
  if norm > 0:
    X = np.minimum(norm, X)
    X = np.maximum(-norm, X)
  return X

def L2(params):
  pass

def save_params(filehead, params):
  for p in params:
    filename = filehead + str(p) 
    np.savetxt(filename, params[p], fmt = "%10.5f", delimiter = ",")
 
def update_params_sgd(params, grads, lr):
  for p in params:
    params[p] -= lr * grads[p]
  return params
      
def update_params_adam(params, grads, lr,
                  m_W1, m_W3, m_b1, m_b3,
                  n_W1, n_W3, n_b1, n_b3,
                  #m_W1, m_W2, m_W3, m_b1, m_b2, m_b3,
                  #n_W1, n_W2, n_W3, n_b1, n_b2, n_b3,
		  alpha_W, beta_W, alpha_b, beta_b):
  eps = 1e-8
  m_W1 = alpha_W * m_W1 + (1 - alpha_W) * grads['W1']
  n_W1 = beta_W * n_W1 + (1 - beta_W) * grads['W1']**2
  params['W1'] -= lr * m_W1 / (np.sqrt(n_W1) + eps)
  m_b1 = alpha_b * m_b1 + (1 - alpha_b) * grads['b1']
  n_b1 = beta_b * n_b1 + (1 - beta_b) * grads['b1']**2
  params['b1'] -= lr * m_b1 / (np.sqrt(n_b1) + eps)
  '''
  m_W2 = alpha_W * m_W2 + (1 - alpha_W) * grads['W2']
  n_W2 = beta_W * n_W2 + (1 - beta_W) * grads['W2']**2
  params['W2'] -= lr * m_W2 / (np.sqrt(n_W2) + eps)
  m_b2 = alpha_b * m_b2 + (1 - alpha_b) * grads['b2']
  n_b2 = beta_b * n_b2 + (1 - beta_b) * grads['b2']**2
  params['b2'] -= lr * m_b2 / (np.sqrt(n_b2) + eps)
  '''
  m_W3 = alpha_W * m_W3 + (1 - alpha_W) * grads['W3']
  n_W3 = beta_W * n_W3 + (1 - beta_W) * grads['W3']**2
  params['W3'] -= lr * m_W3 / (np.sqrt(n_W3) + eps)
  m_b3 = alpha_b * m_b3 + (1 - alpha_b) * grads['b3']
  n_b3 = beta_b * n_b3 + (1 - beta_b) * grads['b3']**2
  params['b3'] -= lr * m_b3 / (np.sqrt(n_b3) + eps)
  return params

# for autoencoder, X and y is the same
def get_next_batch(all_data, batch_size, pos, start_pos, epoch_size):
  if pos - start_pos == epoch_size:
    pos = start_pos
  batch = all_data[:,pos:pos+batch_size]
  return batch.T, batch.T
 
def get_ratings(path, TRAIN_PROB, NOISE_TRAIN_PROB, VALID_PROB, TEST_PROB):
  # create user list and movie list
  # store all input data for later use
  users = []
  movies = []
  ratings = []
  user_to_idx = {}
  movie_to_idx = {}
  with open(path, 'r') as rating_input:
    for line in rating_input:
      words = line.split(',')
      if words[0] == 'userId':
        continue
      users.append(words[0].strip())
      movies.append(words[1].strip())
      ratings.append(words[2].strip())
      if words[0] not in user_to_idx:
        user_to_idx[words[0]] = len(user_to_idx)
      if words[1] not in movie_to_idx:
        movie_to_idx[words[1]] = len(movie_to_idx)
  # create mapping between index and users/movies
  idx_to_user = np.zeros(len(user_to_idx), dtype='int')
  idx_to_movie = np.zeros(len(movie_to_idx), dtype='int')
  for u in user_to_idx:
    idx_to_user[user_to_idx[u]] = u
  for m in movie_to_idx:
    idx_to_movie[movie_to_idx[m]] = m
  
  all_ratings = np.zeros((len(user_to_idx), len(movie_to_idx)))
  train_ratings = np.zeros((len(user_to_idx), len(movie_to_idx)))
  noise_train_ratings = np.zeros((len(user_to_idx), len(movie_to_idx)))
  validate_ratings = np.zeros((len(user_to_idx), len(movie_to_idx)))
  test_ratings = np.zeros((len(user_to_idx), len(movie_to_idx)))
  for i in range(len(users)):
    user = users[i]
    movie = movies[i]
    rating = ratings[i]
    if all_ratings[user_to_idx[user], movie_to_idx[movie]] != 0:
        print 'Duplicated input.'
    all_ratings[user_to_idx[user], movie_to_idx[movie]] = rating
    r = np.random.uniform(0,1)
    if r < TRAIN_PROB:
      train_ratings[user_to_idx[user], movie_to_idx[movie]] = rating
      continue
    elif r < TRAIN_PROB + NOISE_TRAIN_PROB:
      noise_train_ratings[user_to_idx[user], movie_to_idx[movie]] = rating
      continue
    elif r < TRAIN_PROB + NOISE_TRAIN_PROB + VALID_PROB: 
      validate_ratings[user_to_idx[user], movie_to_idx[movie]] = rating
      continue
    else:
      test_ratings[user_to_idx[user], movie_to_idx[movie]] = rating
  #return all_ratings, idx_to_movie, idx_to_user
  return all_ratings, train_ratings, noise_train_ratings, validate_ratings, test_ratings 
