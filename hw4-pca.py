import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.decomposition import PCA

np.random.seed(0)

#############################
# Read .csv and .txt files #
#############################
def read_csv(path = 'co_occur.csv'):
  """
  Reads a .csv file.

  Input:
      - path: Path of the .csv file

  Return:
      - data: A 10,000 x 10,000 numpy array (the covariance matrix)
  """
  
  data = []
  
  with open(path, 'r') as file:
      
    csvreader = csv.reader(file)
    
    for row in csvreader:
      a = []
      
      for el in row:
        a.append(float(el))
        
      a = np.array(a)
      data.append(a)
      
    data = np.array(data)
    print(np.shape(data))
    
  return data


def read_txt(path):
  """
  Reads a .txt file.

  Input:
      - path: Path of the .txt file

  Return:
      - words: A list of strings (i^th item is the i^th row from the file)
  """
    
  with open(path) as f:
     
    data = f.read()
    words = data.split("\n")
  
    print(len(words))
 
  return words


##################################
# Plot Eigenvalues (For Part 1) #
##################################
def plot_evs(D):
  '''
  Input: 
     - D: A sequence of eigenvalues
  
  Plots the eigenvalues against their indices and save
  the figure in 'hw4/ev_plot.png'
  '''
  n = np.shape(D)[0]
  x = range(1, n+1)
  fig, ax = plt.subplots()
  ax.plot(x, D)

  ax.set(xlabel = 'Index', ylabel = 'Eigenvalue')
  fig.savefig("hw4/ev_plot.png")
  plt.show()


#################################################################################
# Find the Embedding of a Given Word (e.g. 'man', 'woman') (For Parts 2, 4, 5) #
#################################################################################
def find_embedding(word, words, E):
  '''
  Inputs: 
      - word: The query word (string)
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
  Return:
      - emb: The word embedding
  '''  
  ##################################################################
  # TODO: Implement the following steps:
  # i) Find the index/position of 'word' in 'words'.
  # ii) The row of 'E' at this index will be the embedding 'emb'. 
  ##################################################################
  idx = np.where(words == word)[0][0]
  emb = E[idx]
  return emb


############################################################
# Find the Word Most Similar to a Given Word (For Part 2) #
############################################################
def find_most_sim_word(word, words, E): 
  '''
  Inputs:
      - word: The query word (string)
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
  Return:
      - most_sim_word: The word which is most similar to the input 'word'
  '''
  ###############################################################################################################
  # TODO: Implement the following steps: 
  # i) Get the word embedding of 'word' using the 'find_embedding' function. 
  # ii) Compute the similarity (dot product) of this embedding with every embedding, i.e. with each row of 'E'. 
  # iii) Find the index of the row of 'E' which gives the largest similarity, exculding the row which is the
  # embedding of 'word' from the search (for this, you will need to find the index/position of 'word' in 'words'). 
  # iv) Find the word corresponding to that index from 'words'.
  ###############################################################################################################
  emb = find_embedding(word, words, E) #(100,)
  sims = E@emb #(10000,) # cosine similarity = dot product (rows of E already have unit norm)

  idx = np.where(words==word)[0][0] #arr, first index
  sims[idx] = -np.inf # exclude itself

  most_sim_word = words[np.argmax(sims)]
  return most_sim_word


#############################################################
# Interpret Principal Components/Eigenvectors (For Part 3) #
#############################################################
def find_info_ev(v, words, k=20): 
  # Find words corresponding to the 'k' largest magnitude elements of an eigenvector,
  # to see what kind of information is captured by 'v'.
  '''
  Inputs: 
      - v: An eigenvector (with 10,000 entries)
      - words: The list of all words read from dictionary.txt
      - k: The number of words the function should return
  Return:
      - info: The set of k words 
  '''
  #########################################################################################################################
  # TODO: Implement the following steps:
  # i) Obtain a set of indices (postions in 'v') which would sort the entries of 'v' in decreasing order of absolute value.
  # ii) Using the set of first 'k' indices, get the corresponding words from the list 'words'.
  ##########################################################################################################################
  # sort decreasing order abs value
  idx = np.argsort(np.abs(v))[::-1]
  info = words[idx[:k]] # select top-k
  return info


#################################################################################
# Explore Semantic/Syntactic Concepts Captured by Word Embeddings (For Part 4) #
#################################################################################

def plot_projections(word_seq, proj_seq, filename = 'hw4/projections.png'):
  '''
  Inputs: 
   - word_seq: A sequence of words (strings), used as labels.
   - proj_seq: A sequence of projection values.
   - filename: The plot will be saved 
  
  Plot the set of values in 'proj_seq' on a line with the corresponding labels from 'word_seq and
  save the figure
  '''
  y = np.zeros(len(proj_seq))
  fig, ax = plt.subplots()
  ax.scatter(proj_seq, y)
  
  # to avoid overlaps in the plot
  plt.rcParams.update({'font.size': 9})
  idx = np.argsort(proj_seq)
  proj_seq = proj_seq[idx]
  word_seq = word_seq[idx]
  del_y_prev = 0
  
  for i, label in enumerate(word_seq):
    # to avoid overlaps in the plot
    if i<len(proj_seq)-1:
      if np.abs(proj_seq[i]-proj_seq[i+1])<0.02:
          del_y0 = -0.005-0.0021*len(label)
          if del_y_prev == 0:
            del_y = del_y0
          else:
            del_y = 0
          del_y_prev = del_y0  
      elif (del_y_prev!=0 and del_y == 0 and np.abs(proj_seq[i]-proj_seq[i-1])<0.02):
          del_y = -0.005-0.0021*len(label)
          del_y_prev = del_y
      else:
          del_y = 0
          del_y_prev = 0
          
    ax.text(x = proj_seq[i]-0.01, y = y[i]+0.005+del_y, s = label, rotation = 90)
  
  ax.set_xlim(-0.5,0.5)

  fig.savefig(filename)
  plt.show()  


def get_projections(word_seq, words, E, w):
  '''
  Inputs: 
      - word_seq: A sequence of words (strings)
      - word: The query word (string)
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
      - w: An embedding vector (with 10,000 elements)
  Return:
      - proj_seq: The sequence of projections of the words in 'word_seq'
  '''
  ###############################################################################################################
  # TODO: Implement the following steps: 
  # i) For each word in 'word_seq':
  #     - Get its word embedding using the 'find_embedding' function. 
  #     - Find the projection of this embedding onto 'w', i.e. the dot product with normalized 'w'. 
  # ii) Return the set of projections. 
  ###############################################################################################################  
  w_norm = w / np.linalg.norm(w)          # unit vector
  proj_seq = []
  for word in word_seq:
      emb = find_embedding(word, words, E)
      proj = np.dot(emb, w_norm)
      proj_seq.append(proj)
  proj_seq = np.array(proj_seq) 
  return proj_seq
    

#####################################################################
# BONUS: Finding Answers to Analogy Questions (For Part 5 (Bonus)) #
#####################################################################
def find_most_sim_word_w(w, words, E, ids):
  # Find the word whose embedding is most similar to a given vector in the embedding space.
  # Similar to the 'find_most_sim_word' function.
  '''
  Inputs: 
      - w: The query vector (with 100 entries) [for an analogy question, this is a combination of three
                            word embeddings (say words wd1, wd2, wd3), as mentioned in the problem statement]
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
      - ids: The indices of words (wd1, wd2, wd3) which need to be excluded from the search while finding the most similar word
  Return:
      - most_sim_word: The word whose embedding which is most similar to the input 'w'
  '''  
  ########################################################################################################################################
  # TODO: Implement the following steps: 
  # i) Compute the similarity (dot product) of 'w' with every embedding, i.e. with each row of 'E'. 
  # ii) Find the index of the row of 'E' which gives the largest similarity, exculding the rows at indices given by 'ids' from the search. 
  # iii) Find the word corresponding to that index from 'words'.
  #########################################################################################################################################
  sims = E @ w # dot products (E rows have unit norm, w is unit)
  for i in ids:
      sims[i] = -np.inf  # exclude given words
  idx = np.argmax(sims)
  most_sim_word = words[idx]      
  return most_sim_word

  
def find_analog(wd1, wd2, wd3, words, E):
  '''
  Inputs: 
      - wd1, wd2, wd3: Three words (strings) posing an analogy question (e.g. for the one in the problem statement, 
                                                                      these are 'man', 'woman', 'king')
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
  Return:
      - wd4: The word whose embedding which is most similar to the input 'w'
  '''
  ########################################################################################################################################
  # TODO: Implement the following steps: 
  # i) Find the word embeddings of wd1, wd2 and wd3 using the 'find_embedding' function: w1, w2, w3, respectively.
  # ii) Get vector w = w2 - w1 + w3, and normalize it. 
  # iii) Find the indices/positions of wd1, wd2, wd3 from the list 'words'.
  # iv) Find wd4, the closest word to w (excluding the given three words from the search by using the indices from the previous step), 
  # using the 'find_most_sim_word_w' function. This will be the answer to the analogy question.
  #########################################################################################################################################
  w1 = find_embedding(wd1, words, E)
  w2 = find_embedding(wd2, words, E)
  w3 = find_embedding(wd3, words, E)

  w = w2 - w1 + w3
  w = w / np.linalg.norm(w) 

  id1 = np.where(words == wd1)[0][0]
  id2 = np.where(words == wd2)[0][0]
  id3 = np.where(words == wd3)[0][0]
  ids = [id1, id2, id3]

  wd4 = find_most_sim_word_w(w, words, E, ids)  
  return wd4


def check_analogy_task_acc(task_words, words, E):
  # Check the accuracy for the analogy task by finding answers to a set of analogy questions #
  '''
  Inputs: 
      - task_words: The list of word sequences (wd1, wd2, wd3, wd4), for an analogy statement (e.g. for 
                     the one in the problem statement, these will be 'man', 'woman', 'king', queen)
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
  Returns:
      - acc: The average accuracy for the task
  '''
  
  acc = 0
  
  for word_seq in task_words:
    word_seq = word_seq.split()
    wd4_ans = find_analog(word_seq[0], word_seq[1], word_seq[2], words, E)
    if wd4_ans == word_seq[3]:
      acc+=1
  acc = acc/len(task_words)
      
  return acc


################################
# Putting Everything Together #
################################

###########################################################################
# Part 0: Read the files (Make sure the filenames and paths are correct) #
###########################################################################
# M: The co-occurrance matrix
# words: The set of all words in 'dictionary.txt'
# task_words: The set of analogy queries from 'analogy_task.txt' (for the BONUS part)
M = read_csv("hw4/co_occur.csv")#(10000,10000)
words = read_txt("hw4/dictionary.txt")#10000
words = np.array(words)
task_words = read_txt('hw4/analogy_task.txt') #5585


###########################################################
# Part 1: Applying PCA to get the low rank approximation #
###########################################################
# TODO:
# i) Apply the log transformation mentioned in the problem statement.
# ii) Get centered M, denoted as Mc. First obtain the (d-dimensional) mean feature vector by averaging across all datapoints (rows).
# Then subtract it from all the n feature vectors.
# iii) Use the PCA function (fit method) from the sklearn library to apply PCA on Mc and get its rank-100 approximation (Go through 
# the documentation available at: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
# iv) Get the set of eigenvalues and obtain the plot using the 'plot_evs' function provided above.

# def check(name, X):
#     print(f"\n==== {name} ====")
#     print("shape:", X.shape)
#     print("NaN:", np.isnan(X).any())
#     print("inf:", np.isinf(X).any())
#     print("max:", np.max(X))
#     print("min:", np.min(X))
# i) log transformation
M_trans = np.log(1 + M)
# check("M_trans", M_trans)

# ii) centering
M_mean = np.mean(M_trans, axis=0) # (d,) (10000,)
# check("M_mean", M_mean)
Mc = M_trans - M_mean # (n, d) (10000,10000)
# check("Mc", Mc)
# print(f"M_trans shape: {M_trans.shape}")
# print(f"M_mean shape: {M_mean.shape}")
# print(f"Mc shape: {Mc.shape}")
# iii) PCA
pca = PCA(n_components=100)
# pca = PCA(n_components=100, svd_solver='randomized')
# pca = PCA(n_components=100, svd_solver='full')  
pca.fit(Mc)

# iv) eigenvalues 
eigenvalues = pca.explained_variance_

# plot
plot_evs(eigenvalues)
print("Variance explained by first 100 eigenvalues: ", np.sum(pca.explained_variance_ratio_))

##########################################################
# Part 2: Finding the word most similar to a given word #
##########################################################
# TODO:
# i) Get the set of eigenvectors, i.e. the matrix V (size 10,000 x 100). This will be used in Parts 2, 3.
# Note: Make sure the dimensions are correct (apply transpose, if required).
# ii) Get P = McV. Normalize the columns of P (to have unit l2-norm) to get E. This will be used in Parts 2, 4, 5.
# iii) Normalize the rows of E to have unit l2-norm.
# iv) Complete the 'find_embedding' function and the 'find_most_sim_word' function. 
# v) Pick some word(s), e.g. 'university', uqertyse the 'find_most_sim_word' function to find the most similar word.
V = pca.components_.T   # (10000, 100)

# ii) P = Mc @ V,  then normalize columns of P
P = Mc @ V
col_norms = np.linalg.norm(P, axis=0, keepdims=True)   # (1, 100)
col_norms = np.where(col_norms == 0, 1, col_norms)
E = P / col_norms              # normalize each column (PC direction)

# iii) Normalize rows of E to unit l2-norm
row_norms = np.linalg.norm(E, axis=1, keepdims=True)
row_norms = np.where(row_norms == 0, 1, row_norms)
E_final = E / row_norms          # (10000, 100)  — final embedding matrix

# iv-v) Find similar words
for query in ['learning', 'university', 'california']:
    sim = find_most_sim_word(query, words, E_final)
    print(f"{query}: {sim}")


#########################################################################################
# Part 3: Looking at the information captured by the principal components/eigenvectors #
#########################################################################################
# TODO:
# i) Complete the 'find_info_ev' function, which finsd the words corresponding to k largest magnitude
# elements of a given eigenvector.
# ii) Choose a set of eigenvectors (i.e. columns) from V.
# iii) For each eigenvector, use the 'find_info_ev' function to see what kind of information it captures. Print the results.

print("\n--- PC interpretation ---\n")
for i in [0, 1, 2, 3, 4]:
    v = V[:, i]
    top_words = find_info_ev(v, words, k=10)
    print(f"PC {i+1}: {top_words}")

print("\n--- Last few PCs (small eigenvalues) ---\n")
for i in [95, 96, 97, 98, 99]:
    v = V[:, i]
    top_words = find_info_ev(v, words, k=10)
    print(f"PC {i+1}: {top_words}")

# print("V NaN:", np.isnan(pca.components_).any())
# print("V inf:", np.isinf(pca.components_).any())
# print("P NaN:", np.isnan(P).any())
# print("P inf:", np.isinf(P).any())
# print("E NaN:", np.isnan(E_final).any())
# print("E inf:", np.isinf(E_final).any())
##############################################################################
# Part 4: Exploring Semantic/Syntactic Concepts Captured by Word Embeddings #
##############################################################################
# TODO:
# i) Use the 'find_embedding' function to find w1 and w2 (the embeddings for 'man' and 'woman', respectively). 
# ii) Get w = w1 - w2 and normalize it.
# iii) Complete the 'get_projections' function.
# iv) Use this function to get the set of projections for the list of words mentioned in Part 4.1. 
# Use the 'plot_projections' function to obtain the plot of projections with their corresponding word labels.
# v) Repeat the previous step for the list of words mentioned in Part 4.2 to generate a second plot. You can use the 
# 'filename' argument to make sure the 'plot_projections' function does not overwrite the existing file to save the new plot.
w1 = find_embedding('woman', words, E_final)
w2 = find_embedding('man',   words, E_final)
w = w1 - w2
w_hat = w / np.linalg.norm(w)

# Part 4.1
word_seq = np.array(['boy', 'girl', 'brother', 'sister', 'king', 'queen', 'he', 'she','john', 'mary', 'wall', 'tree'])
proj = get_projections(word_seq, words, E_final, w_hat)
plot_projections(word_seq, proj, filename='hw4/4.4.1.png')

# Part 4.2
word_seq = np.array(['math', 'history', 'nurse', 'doctor', 'pilot', 'teacher', 'engineer', 'science', 'arts', 'literature', 'bob', 'alice'])
proj = get_projections(word_seq, words, E_final, w_hat)
plot_projections(word_seq, proj, filename='hw4/4.4.2.png')

###########################################################
# Part 5 (BONUS): Finding answers to analogy questions #
###########################################################
# TODO:
# i) Complete the 'find_analog' function.
# ii) Test it using a set of words for an analogy question, such as the one given in the problem statement.
# iii) Use the 'check_analogy_task_acc' function to evaluate 'find_analog' on the list of analogy queries in 'task_words'.

# check_analogy_task_acc
acc = check_analogy_task_acc(task_words, words, E_final)
print(f"Analogy Task Accuracy: {acc * 100:.2f}%")

# find_analog
correct_ex, wrong_ex = [], []
for line in task_words:
    parts = line.split()
    if len(parts) != 4:
        continue
    wd1, wd2, wd3, wd4 = parts
    pred = find_analog(wd1, wd2, wd3, words, E_final)
    
    if pred == wd4:#correct
        correct_ex.append((wd1, wd2, wd3, wd4, pred))
    else:#wrong
        wrong_ex.append((wd1, wd2, wd3, wd4, pred))

print("\nCorrect examples (first 5):")
for ex in correct_ex[:5]:
    print(f"  {ex[0]}:{ex[1]} :: {ex[2]}:{ex[3]} → {ex[4]}")

print("\nWrong examples (first 5):")
for ex in wrong_ex[:5]:
    print(f"  {ex[0]}:{ex[1]} :: {ex[2]}:{ex[3]} → {ex[4]}")
