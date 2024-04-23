import numpy as np
class LinearRegression:
    def __init__(self,lr=.001,n_iter=1000):
        self.lr=lr
        self.n_iter=n_iter
        self.weights=None
        self.bias=None

    def fit(self,x,y):
        n_samples,n_features = x.shape
        #initialze weights and bias
        self.weights=np.random.rand(n_features,1)
        self.bias=0
        for _ in range(self.n_iter):
            y_predicted = np.dot(x, self.weights).flatten() + self.bias

            #compute gradients
            dw=(1/n_samples)*np.dot(x.T,(y_predicted-y).reshape(-1,1))
            db=(1/n_samples)*np.sum(y_predicted-y)
            self.weights-=self.lr*dw
            self.bias-=self.lr*db
    def predict(self,x):
        y_predicted=np.dot(x,self.weights).flaten()+self.bias
        return y_predicted