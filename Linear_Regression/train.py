import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression 
x,y=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1234)
reg=LinearRegression(lr=0.01)
reg.fit(x_train,y_train)
predictions=reg.predict(x_test)
def mse(y_test,predictions):
    return np.mean((y_test-predictions)**2)
mse=mse(y_test,predictions)
print(f'MSE : {mse}')
y_pred_line=reg.predict(x)
cmp=plt.get_cmap('viridis')
fig=plt.figure(figsize=(0,6))
m1 = plt.scatter(x_train, y_train, c='blue', s=10, label='Trained points')

m2 = plt.scatter(x_test, y_test, c='red', s=10, label='Test points')

plt.plot(x,y_pred_line,color='black',linewidth=2,label='fitted line')
plt.legend()
plt.show()


