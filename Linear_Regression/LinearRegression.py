import numpy as np

# create linear algebra model
class LinearRegression:
    # constructor for linear regression attributes that i'll use in the model
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr #learing rate
        self.n_iters=n_iters # number of uterations that i'll do on the data 
        # the eq of linear regression model wX+b
        self.wieghts=None    # w
        self.bias=None       # b


    # use the traing data to fit the line on it
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

    # use the test data to predict the output
    def predict(self,x):
        y_pred=np.dot(x,self.wieghts)+self.bias
        return y_pred
