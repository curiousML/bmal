from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

class PassiveLearning:
    def __init__(self, X_train_full, y_train_full, full_id, train_id, pool_id, b):
        self.X = X_train_full
        self.y = y_train_full
        self.full_id = full_id
        self.train_id = train_id
        self.pool_id = pool_id
        
        self.perfs = []
        self.perfs_f1 = []
        self.t = 0
        self.K = None
        self.b = b
        self.n = len(self.full_id)
        self.nl = len(self.train_id)
        self.nu = len(self.pool_id)
        
        self.model = LogisticRegression(multi_class='auto', penalty='l2', solver='saga', tol=0.01, class_weight = "balanced")
        self._train()
        
    def _train(self):
        self.model.fit(self.X[self.train_id], self.y[self.train_id])
        
    def performance_auc(self, X_test, y_test):
        score = roc_auc_score(y_test, self.model.predict_proba(X_test)[:,1])
        return(score)
    
    def performance_f1score(self, X_test, y_test):
        score = f1_score(y_test, self.model.predict(X_test))
        return(score)
    
    def ite(self):
        self.selec_id = np.random.choice(self.pool_id, self.b)
        self.t = self.t + 1
        # update pool and train sets
        self.pool_id = self.pool_id[~np.isin(self.pool_id, self.selec_id)]
        self.train_id = np.concatenate([self.train_id, self.selec_id])
        
        # training
        self._train()
        
        return(self.selec_id)
    
    def run(self, X_test, y_test, times = 10):
        self.perfs.append(self.performance_auc(X_test, y_test))
        self.perfs_f1.append(self.performance_f1score(X_test, y_test))
        for t in range(times):
            self.selec_id = self.ite()
            self.perfs.append(self.performance_auc(X_test, y_test))
            self.perfs_f1.append(self.performance_f1score(X_test, y_test))
        return self.perfs_f1
        #return self.perfs

