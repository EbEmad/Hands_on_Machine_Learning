import numpy as np

class LinearRegression:
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.wieghts=None
        self.bias=None
    def fit(self,x,y):
        n_samples,n_features=x.shape
        self.wieghts=np.zeros(n_features)
        self.bias=0
        for _ in range(self.n_iters):
            y_pred=np.dot(x,self.wieghts)+self.bias
            dw=(1/n_samples)*np.dot(x.T,y_pred-y)
            db=(1/n_samples)*np.sum(y_pred-y)
            self.wieghts=self.wieghts-self.lr*dw
            self.bias=self.bias-self.lr*db

    def predict(self,x):
        y_pred=np.dot(x,self.wieghts)+self.bias
        return y_pred
