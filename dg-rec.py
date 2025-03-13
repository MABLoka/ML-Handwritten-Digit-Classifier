import cv2
import pickle
import numpy as np
from itertools import combinations
import threading
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
i = 0
# Global variables for threading
prediction_result = None
prediction_lock = threading.Lock()
prediction_event = threading.Event()

def preprocess_and_predict(img):
    """Preprocess the image and make an SVM prediction."""
    global prediction_result

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    img_bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    # img_aux = cv2.resize(img_bw, (200, 200))
    
    img_resized = cv2.resize(img_bw, (28, 28))  # Resize to MNIST size
    img_inverted = cv2.bitwise_not(img_resized)
    img_defined = np.array(img_inverted, dtype='float')
    # cv2.imwrite('bw_image.png', img_inverted)
    img_flattened = img_defined.flatten().reshape(1, -1)

    # Normalize
    img_flattened = img_flattened / 255.0
    # print(img_inverted.shape)
    # img = np.linalg.inv(img)
    # Predict in a separate thread
    
    prediction = loaded_svm.predict(img_flattened)
    
    with prediction_lock:
        prediction_result = prediction[0]
        prediction_event.set()  # Signal that a new prediction is ready

def prediction_thread_function():
    """Continuously predict on the latest frame in the background."""
    while True:
        prediction_event.wait()  # Wait for the main thread to provide a new image
        prediction_event.clear()  # Reset event for the next iteration


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

#load trained SVM model
with open("multi_svm_model.pkl", "rb") as f:
    loaded_svm = pickle.load(f)

while True:
    
    _, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    bbox_size = (100, 100)
    bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
             (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))]
    
    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img_g = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_g = cv2.bitwise_not(img_g)
    
    if not prediction_event.is_set():
        threading.Thread(target=preprocess_and_predict, args=(img_cropped,), daemon=True).start()

    # Display latest prediction
    with prediction_lock:
        if prediction_result is not None:
            cv2.putText(frame_copy, f"Prediction: {prediction_result}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            

    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 3)
    

    cv2.imshow("input", frame_copy)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()