import numpy as np

class SVMc:
    def __init__(self, Lambda, C, kernel, learningRate = .001, n_iter = 1000):

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
        pass

    self predict()
