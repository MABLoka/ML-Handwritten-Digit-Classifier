# for part 4.2, write your code here
from itertools import combinations
from datasets import load_dataset
import numpy as np
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
import cv2

trainset = load_dataset('mnist', split='train')
train_data = trainset['image']
train_label = trainset['label']

testset = load_dataset('mnist', split='test')
test_data = testset['image']
test_label = testset['label']

train_data = np.array(train_data, dtype='float')/255 # norm to [0,1]
train_data = np.reshape(train_data,(60000,28*28))
train_label = np.array(train_label, dtype='short')
test_data = np.array(test_data, dtype='float')/255 # norm to [0,1]
test_data = np.reshape(test_data,(10000,28*28))
test_label = np.array(test_label, dtype='short')

print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

# prepare digits '1' and '7' for binary SVMs

# digit_train_index = np.isin(train_label,[1,5,7,8,9])
# X_train = train_data[digit_train_index]
# y_train = train_label[digit_train_index]
# digit_test_index = np.isin(train_label,[1,5,7,8,9])
# X_test = test_data[digit_test_index]
# y_test = test_label[digit_test_index]

selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
digit_train_index = np.isin(train_label, selected_digits)
print(digit_train_index)
X_train = train_data[digit_train_index]

y_train = train_label[digit_train_index]
digit_train_index = np.isin(test_label, selected_digits)
X_test = test_data[digit_train_index]
y_test = test_label[digit_train_index]
print(X_test)
# normalize all feature vectors to unit-length
X_train = np.transpose (X_train.T / np.sqrt(np.sum(X_train*X_train, axis=1)))
X_test =  np.transpose (X_test.T  / np.sqrt(np.sum(X_test*X_test, axis=1)))

# # convert labels: '1' => -1, '7' => +1
# CUTOFF = 6 # any number between '1' and '7'
# y_train = np.sign(y_train-CUTOFF)
# y_test = np.sign(y_test-CUTOFF)

class mySVM2():
  def __init__(self, kernel='linear', optimizer='pgd', debug=0, threshold=0.001, \
               lr=1.0, max_epochs=20, batch_size=2, C=1, order=3, gamma=1.0):
    self.kernel = kernel           # kernel type
    self.optimizer = optimizer     # which optimizer is used to solve quadratic programming
    self.lr = lr                   # max learning rate in PGD
    self.max_epochs = max_epochs   # max epochs in PGD
    self.batch_size = batch_size   # size of each subset in PGD
    self.debug = debug             # whether print debugging info
    self.threshold = threshold     # threshold to filter out support vectors

    self.C = C                     # C for the soft-margin term
    self.order = order             # power order for polynomial kernel
    self.gamma = gamma             # gamma for Gaussian RBF kernel

  # Kernel Function
  # X[N,d]: training samples;  Y[M,d]: other training samples
  # return Q[N,N]: linear kernel matrix between X and Y
  def Kernel(self, X, Y):
    if (self.kernel == 'linear'):
      K = X @ Y.T
    elif (self.kernel == 'poly'):
      K = np.power(X @ Y.T +1, self.order)
    elif (self.kernel == 'rbf'):
      d1 = np.sum(X*X, axis=1)
      d2 = np.sum(Y*Y, axis=1)
      K = np.outer(d1, np.ones(Y.shape[0])) + np.outer(np.ones(X.shape[0]), d2) \
          - 2 * X @ Y.T
      K = np.exp(-self.gamma * K)

    return K

  # construct matrix Q from any kernel function for dual SVM optimization
  def QuadraticMatrix(self, X, y):
    Q = np.outer(y, y) * self.Kernel(X, X)
    return Q

  # use projected gradient descent to solve quadratic program
  # refer to Algorithm 6.5 on page 127
  # Q[N,N]: quadratic matrix;  y[N]: training labels (+1 or -1)
  def PGD(self, Q, y):
    N = Q.shape[0]   # num of training samples
    alpha = np.zeros(N)
    prev_L = 0.0

    for epoch in range(self.max_epochs):
      indices = np.random.permutation(N)  #randomly shuffle data indices
      for batch_start in range(0, N, self.batch_size):
        idx = indices[batch_start:batch_start + self.batch_size] # indices of the current subset
        alpha_s = alpha[idx]
        y_s = y[idx]

        grad_s = Q[idx,:] @ alpha - np.ones(idx.shape[0])
        proj_grad_s = grad_s - np.dot(y_s,grad_s)/np.dot(y_s, y_s)*y_s

        bound = np.zeros(idx.shape[0])
        bound[proj_grad_s < 0] = self.C

        eta = np.min(np.abs(alpha_s-bound)/(np.abs(proj_grad_s)+0.001))

        alpha[idx] -= min(eta, self.lr) * proj_grad_s

      L = 0.5 * alpha.T @ Q @ alpha - np.sum(alpha) # objectibve function
      if (L > prev_L):
        if (self.debug>0):
          print(f'Early stopping at epoch={epoch}! (reduce learning rate lr)')
        break

      if (self.debug>1):
        print(f'[PGD optimizer] epoch = {epoch}: L = {L:.5f}  (# of support vectors = {(alpha>self.threshold).sum()})')
        print(f'                 alpha: max={np.max(alpha)} min={np.min(alpha)} orthogonal constraint={np.dot(alpha,y):.2f}')

      prev_L = L

    return alpha

  # train SVM from training samples
  # X[N,d]: input features;  y[N]: output labels (+1 or -1)
  def fit(self, X, y):
    if(self.kernel != 'linear' and self.kernel != 'poly' and self.kernel != 'rbf'):
      print("Error: only linear/poly/rbf kernel is supported!")
      return

    Q = self.QuadraticMatrix(X, y)

    alpha = self.PGD(Q, y)

    #save support vectors (pruning all data with alpha==0)
    self.X_SVs = X[alpha>self.threshold]
    self.y_SVs = y[alpha>self.threshold]
    self.alpha_SVs = alpha[alpha>self.threshold]

    if(self.kernel == 'linear'):
      self.w = (self.y_SVs * self.alpha_SVs) @ self.X_SVs

    # estimate b
    idx = np.nonzero(np.logical_and(self.alpha_SVs>self.threshold,self.alpha_SVs<self.C-self.threshold))
    if(len(idx) == 0):
      idx = np.nonzero(self.alpha_SVs>self.threshold)
    # refer to the formula on page 125 (above Figure 6.11)
    b = self.y_SVs[idx] - (self.y_SVs * self.alpha_SVs) @ self.Kernel(self.X_SVs, self.X_SVs[idx])
    self.b = np.median(b)

    return

  # use SVM from prediction
  # X[N,d]: input features
  def predict(self, X):
    
    if(self.kernel != 'linear' and self.kernel != 'poly' and self.kernel != 'rbf'):
      print("Error: only linear/poly/rbf kernel is supported!")
      return

    if(self.kernel == 'linear'):
      y = X @ self.w + self.b
    else:
      y = (self.y_SVs * self.alpha_SVs) @ self.Kernel(self.X_SVs, X) + self.b

    return np.sign(y)

class MultiClassSVM():
    def __init__(self, C=1, kernel='rbf', gamma='scale', debug=0, max_epochs=20, lr=1.0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.debug = debug
        self.max_epochs = max_epochs
        self.lr = lr
        self.models = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        pairs = list(combinations(self.classes, 2))
        for (class1, class2) in pairs:
            idx = np.logical_or(y == class1, y == class2)
            X_pair = X[idx]
            y_pair = y[idx]
            y_pair = np.where(y_pair == class1, -1, 1)

            model = mySVM2(max_epochs=self.max_epochs, lr=self.lr, C=self.C, kernel=self.kernel, gamma=self.gamma, debug=self.debug)
            model.fit(X_pair, y_pair)
            self.models[(class1, class2)] = model

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.classes)))
        for (class1, class2), model in self.models.items():
          predictions = model.predict(X)
          class1_index = np.where(self.classes == class1)[0][0]
          class2_index = np.where(self.classes == class2)[0][0]
          votes[:, class1_index] += (predictions == -1)
          votes[:, class2_index] += (predictions == 1)

        print("Votes per class:\n", votes[:10])
        return self.classes[np.argmax(votes, axis=1)]



# Train and evaluate the multi-class SVM with majority voting
# multi_svm = MultiClassSVM(C=2, kernel='rbf', gamma=2.0, max_epochs=20, lr=1.0, debug=0)
# multi_svm.fit(X_train, y_train)
# predict = multi_svm.predict(X_train)
# train_acc = np.count_nonzero(np.equal(predict, y_train)) / y_train.size
# predict = multi_svm.predict(X_test)
# test_acc = np.count_nonzero(np.equal(predict, y_test)) / y_test.size
# print(f'Multi-class SVM: training accuracy={100 * train_acc:.2f}% test accuracy={100 * test_acc:.2f}%')

# with open("multi_svm_model.pkl", "rb") as f:
#     loaded_svm = pickle.load(f)

# img = cv2.imread('bw_image.png', cv2.IMREAD_GRAYSCALE)
# # img = np.array(img, dtype='float')
# predict = loaded_svm.predict(img.flatten().reshape(1, -1))

# print(predict)

# fig = plt.figure()
# # img = test_data[700]  #reshape each image from 1x784 to 28x28 for display
# img = np.array(img, dtype='float')
# pixels = img.reshape((28,28))
# ax = fig.add_subplot(3,3,1)
# ax.title.set_text('original')

# plt.imshow(pixels, cmap='gray')
# plt.title(f"Predicted Label: {predict}")
# plt.axis('off')  # Hide axes
# plt.show()


c = 1
g = 0.001
multi_svm = MultiClassSVM(C=c, kernel='rbf', gamma=g, max_epochs=20, lr=1.0, debug=0)
multi_svm.fit(X_train, y_train)
predict = multi_svm.predict(X_train)
train_acc = np.count_nonzero(np.equal(predict, y_train)) / y_train.size
predict = multi_svm.predict(X_test)
test_acc = np.count_nonzero(np.equal(predict, y_test)) / y_test.size
print(f'Multi-class non-linear (RBF) SVM (C={c}, gamma={g}): training accuracy={100 * train_acc:.2f}% test accuracy={100 * test_acc:.2f}%')


# Save the trained model to a file
with open('multi_svm_model.pkl', 'wb') as f:
    pickle.dump(multi_svm, f)

print("Model saved successfully!")

# predict = loaded_svm.predict(test_data[0].reshape(1, -1))

# print(predict)
# fig = plt.figure()
# # img = cv2.open('bw_image.png')  #reshape each image from 1x784 to 28x28 for display
# img = test_data[0]
# pixels = img.reshape((28,28))
# ax = fig.add_subplot(3,3,1)
# ax.title.set_text('original')

# plt.imshow(pixels, cmap='gray')
# plt.title(f"Predicted Label: {predict}")
# plt.axis('off')  # Hide axes
# plt.show()