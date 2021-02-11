from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import logging

class ActiveLearning:
    def __init__(self, X_train_full, y_train_full, full_id, train_id, pool_id, b, K = None, h = None, lam=None):
        self.X = X_train_full
        self.y = y_train_full
        self.full_id = full_id
        self.train_id = train_id
        self.pool_id = pool_id
        self.b = b
        self.lam = lam
        if self.lam is None:
            self.lam = self.b**2 # self.b**4, np.sqrt(self.b)
        self.n = len(self.full_id)
        self.nl = len(self.train_id)
        self.nu = len(self.pool_id)
        
        self.perfs = []
        self.perfs_f1 = []
        self.t = 0
        self.K = K
        self.h = h
        if self.K is None:
            self.compute_kernel() # long
        if self.h is None:
            self.compute_h() # long
        self.h_init = self.h
        self.model = LogisticRegression(multi_class='auto', penalty='l2', solver='saga', tol=0.01, class_weight = "balanced")
        self._train()
        
    def _train(self):
        self.model.fit(self.X[self.train_id], self.y[self.train_id])
        
    def _LC_vect(self, ids, eps = None):
        C = np.max(self.model.predict_proba(self.X[ids]), axis = 1)
        C_epsilon = C
        if eps is None:
            gamma = 20
            tau = 10
            beta = gamma * (self.nu/self.n)**tau
            eps = np.quantile(C, beta/100)
        C_epsilon[C_epsilon < eps] = eps
        return(C_epsilon) # or np.max
    
    def LC_score(self, ids, eps = None):
        C_epsilon = self._LC_vect(ids, eps)
        return(np.sum(C_epsilon)) # or np.max
    
    def compute_kernel(self, p = None):
        self.pairwise_dists = squareform(pdist(self.X, 'euclidean'))
        if p is None:
            p = np.median(self.pairwise_dists ** 2)
        self.K = np.exp(-self.pairwise_dists ** 2 / p)
    
    def R_score(self, ids):
        R = 0.5 * self.K[np.ix_(ids, ids)].sum() + \
        (self.nu - self.b)/self.n * self.K[np.ix_(ids, self.train_id)].sum() + \
        (self.nl + self.b)/self.n * self.K[np.ix_(ids, self.pool_id)].sum()
        return R
    
    def compute_h(self):
        self.h = (self.nu - self.b)/self.n * self.K[np.ix_(self.full_id, self.train_id)].sum(axis = 1) + \
                    (self.nl + self.b)/self.n * self.K[np.ix_(self.full_id, self.pool_id)].sum(axis = 1)

    def compute_psi(self):
        self.psi = 0.5 + self.h[self.pool_id] + self.lam * self._LC_vect(self.pool_id)
        return(self.psi)

    def rand_greedy(self):
        """
        afin d'avoir psi, selec_id
        """
        self.selec_id = []
        self.psi = self.compute_psi()
        for i in range(self.b):
            index = np.argsort(self.psi)[:self.b]
            e_pos = np.random.choice(index)
            e = self.pool_id[~np.isin(self.pool_id, self.selec_id)][e_pos]
            self.selec_id.append(e)
            # mettre à jour les valeurs psi
            psi_prime = np.delete(self.psi, e_pos) - self.K[self.pool_id[~np.isin(self.pool_id, self.selec_id)], e]
            self.psi = psi_prime
        return(self.selec_id)

    def update_h(self):
        self.h = self.h - (self.b/self.n) * self.K[np.ix_(self.full_id, self.full_id)].sum(axis = 1) + \
            ((self.nu - self.nl - 4*self.b)/self.n) * self.K[np.ix_(self.full_id, self.selec_id)].sum(axis = 1)
    
    def ite(self):
        # MBAL iteration
        self.selec_id = self.rand_greedy() # long
        self.update_h()
        self.t = self.t + 1
        
        # update pool and train sets
        self.pool_id = self.pool_id[~np.isin(self.pool_id, self.selec_id)]
        self.train_id = np.concatenate([self.train_id, self.selec_id])
        
        # training
        self._train()
        
        return(self.selec_id)
    
    def performance_auc(self, X_test, y_test):
        score = roc_auc_score(y_test, self.model.predict_proba(X_test)[:,1])
        return(score)
    
    def performance_f1score(self, X_test, y_test):
        score = f1_score(y_test, self.model.predict(X_test))
        return(score)
    
    def run(self, X_test, y_test, times = 10, b_descent=None):
        self.perfs.append(self.performance_auc(X_test, y_test))
        self.perfs_f1.append(self.performance_f1score(X_test, y_test))
        for t in range(times):
            self.selec_id = self.ite()
            self.perfs.append(self.performance_auc(X_test, y_test))
            self.perfs_f1.append(self.performance_f1score(X_test, y_test))
            if b_descent is not None:
                logging.info(f"active learning, b : {self.b}")
                self.b = self.b - b_descent
        return self.perfs