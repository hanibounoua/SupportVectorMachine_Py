import numpy as np

class SVMc:
    def __init__(self, Lambda = .5, C = 1, kernel = 'leaner', learningRate = .001, n_iter = 1000):

        # Model's Hyper-Parameters:
        self.Lambda = Lambda # Ponderation Parameter for on the  Hinge Loss function
        self.C = C # The Cost Parameter of the mis classified points on the Hinge Loss function.
        self.kernel = kernel # Not linear transformation of D
        self.lr = learningRate # learning Rate Hyper-Parameter that control convergence of training algorithms
        self.n_iter = n_iter # Number of iteration on training algorithm

        # Model's Parameters:
        self.omega = None
        self.b = None

    def fit(self, X, y):

        n_sample, n_feature = X.shape
        y_ = np.where(y == 0, -1, 1) # Here I ve used the same nomencalture as I found.
        # On tutorial, So this variable is used to define positive and negative class.
        self.omega = np.zeros(n_feature)
        self.b = 0

        # Gradian Descent ---------------------------------------------------------

        for _ in range(self.n_iter):
            for ind, x in enumerate(X):

                if y_[ind]*(np.dot(self.omega, x) - self.b) >= 1:
                    self.omega -= self.lr * 2 * self.Lambda * self.omega
                else:
                    self.omega -= self.lr * ((2 * self.Lambda * self.omega) - self.C * np.dot(x, y_[ind]))
                    self.b -= self.lr * self.C * y[ind]


    def predict(self, X):
        return np.where(np.signe(np.dot(self.omega, X) - self.b) > 0, 1, 0)
