
import numpy as np
from collections import Counter
def Euclidean_Distance(arr1,arr2):
    distance=np.sqrt(np.sum((arr1-arr2)**2))
    return distance


class KNN:
    def __init__(self, k=3):
        self.k=k
    
    def fit(self,x,y):
        self.x_train=x
        self.y_train=y
    
    def predict(slef,x):
        predictions=[self._predict(X) for X in x]
        return predictions
    
    def _predict(self,x):
        # compute the distances
        distances=[Euclidean_Distance(x,x_train) for x_train in self.x_train]

        # get th closest k
        k_indices=np.argsort(distances)[:self.k]
        k_nearest_labels=[self.y_train[i] for i in k_indices]

        # majority voye
        most_common=Counter(k_nearest_labels).most_common()
        return most_common


        
