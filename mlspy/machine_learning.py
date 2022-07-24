from re import T
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import seaborn as sns
import itertools
from collections import Counter
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class LinearRegression:
    def __init__(self,learning_rate = 0.001,n_iteration=1000):
        if type(learning_rate) == float: 
            self.learning_rate = learning_rate
        if type(n_iteration) == int: 
            self.n_iteration = n_iteration
        else:
            raise ValueError("Please enter a valid value!")
    
    def calculate(self,W):
        return (self.X @ W) + self.b
    
    def calculate_loss(self):
        return np.square(self.Y - self.calculate(self.W)).mean()

    def update_weights(self):
        self.dw = np.zeros([self.features,1])
        self.db = 0
        self.dw = (self.X.T @ (self.calculate(self.W) - self.Y )) /self.features
        self.W -= self.dw * self.learning_rate

        self.db = np.sum((self.calculate(self.W) - self.Y )) /self.features
        self.b -= self.db * self.learning_rate

    def train(self,X,y):
        if X.size == 0 or y.size == 0:
            raise ValueError("Size of array is zero!")
        #if y.shape[1] > 1:
            #raise ValueError("Inputs' should be 1D.")
        if len(X) != len(y):
            raise ValueError("Shape of Input and Target different!")

        self.X = np.array(X)
        self.Y = np.array(y)
        self.Y = self.Y.reshape(-1,1)
        self.b  = 0
        self.N = self.X.shape[0]
        self.features = self.X.shape[1]
        self.W = np.random.randn(self.features,1)
        
        for i in range(self.n_iteration):
            self.update_weights()
            print(f"No of iteration {i}----> Loss {self.calculate_loss()}")
            if self.calculate_loss() < 0.00001:
                break

    def predict(self,X):
        try: 
            return (X @ self.W) + self.b
        except Exception: 
            return print(f"This {self.__class__.__name__} instance is not trained yet. Call 'train' with appropriate arguments before using this estimator.")

class RidgeRegression(LinearRegression):
    def __init__(self,learning_rate=0.001,l2 = 0.2,n_iteration = 100):
        LinearRegression.__init__(self)
        if type(learning_rate) == float: 
            self.learning_rate = learning_rate
        if type(n_iteration) == int: 
            self.n_iteration = n_iteration
        if type(l2) == float:
            self.l2 = l2
        else:
            raise ValueError("Please enter a valid value!")
    
    def update_weights(self):
        self.dw = np.zeros([self.features,1])
        self.db = 0
        
        self.ridge = 2 * self.l2 * self.W
        self.dw = ((self.X.T @ (self.calculate(self.W) - self.Y )) + 2*self.l2*(self.W)) 
        self.dw = self.dw + self.ridge /self.features
        self.W -= self.dw * self.learning_rate
        
        self.db = np.sum((self.calculate(self.W) - self.Y )) /self.features
        self.b -= self.db * self.learning_rate

class LassoRegression(LinearRegression):
    def __init__(self,learning_rate=0.001,l1 = 0.1,n_iteration = 100):
        LinearRegression.__init__(self)
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.l1 = l1
        if type(learning_rate) != float: raise ValueError("Please enter a valid value!")
        if type(n_iteration) != int: raise ValueError("Please enter a valid value!")
        if type(l1) != float: raise ValueError("Please enter a valid value!")

    def update_weights(self):
        self.dw = np.zeros([self.features,1])
        self.db = 0

        self.lasso = self.l1 * np.abs(self.W)
        self.dw = (self.X.T @ (self.calculate(self.W) - self.Y )) + self.lasso * self.l1
        self.dw = self.dw + self.lasso /self.features
        self.W -= self.dw * self.learning_rate
        
        self.db = np.sum((self.calculate(self.W) - self.Y)) /self.features
        self.b -= self.db * self.learning_rate

class ElasticNetRegression(LinearRegression):
    def __init__(self,learning_rate=0.001,l1 = 0.1,l2 = 0.2,n_iteration = 100):
        LinearRegression.__init__(self)
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.l1 = l1
        self.l2 = l2
        if type(learning_rate) != float: raise ValueError("Please enter a valid value!")
        if type(n_iteration) != int: raise ValueError("Please enter a valid value!")
        if type(l1) != float or type(l2) != float: raise ValueError("Please enter a valid value!")

    def update_weights(self):
        self.dw = np.zeros([self.features,1])
        self.db = 0

        self.lasso = self.l1 * np.abs(self.W)
        self.ridge = 2 * self.l2 * self.W
        self.dw = (self.X.T @ (self.calculate(self.W) - self.Y )) + (self.lasso * self.l1)  + self.ridge
        self.dw = self.dw + self.lasso /self.features
        
        self.W -= self.dw * self.learning_rate
        self.db = np.sum((self.calculate(self.W) - self.Y)) /self.features
        self.b -= self.db * self.learning_rate

class LogisticRegression:
    def __init__(self,learning_rate = 0.01,n_iteration = 1000):
        if type(learning_rate) == float: 
            self.learning_rate = learning_rate
        if type(n_iteration) == int: 
            self.n_iteration = n_iteration
        else:
            raise ValueError("Please enter a valid value!")
    
    def train(self,X,y):
        if X.size == 0 or y.size == 0:
            raise ValueError("Size of array is zero!")
        #if y.shape[1] > 1:
            #raise ValueError("Inputs' should be 1D.")
        if len(X) != len(y):
            raise ValueError("Shape of Input and Target different!")

        self.X = np.array(X)
        self.y = np.array(y)
        self.y  = self.y.reshape([-1,1])
        self.N,self.features = self.X.shape
        
        self.classes = np.unique(self.y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        if len(np.unique(self.y)) > 2:
            self.y_onehot = self.one_hot(self.y)
            self.W = np.random.randn(self.features,self.y_onehot.shape[1])
        if len(np.unique(self.y)) == 2:
            self.y_onehot = self.y
            self.W = np.random.randn(self.features,1)
        self.b = 0

        for _ in range(self.n_iteration):
            self.update_weights()
            #print(f"No of iteration {i}----> Loss {self.calculate_loss()}")

    def softmax(self,z):
        return np.exp(z) / np.sum(np.exp(z),axis=1,keepdims=True)
    
    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]
    
    def forward_prop(self,X):
        z = -(X @ self.W) + self.b
        if len(np.unique(self.y)) > 2:
            y_softmax = self.softmax(z)
        else:
            y_softmax = self.sigmoid(z)
        return y_softmax 

    def calculate_loss(self):
        self.y_head = self.forward_prop(self.X)
        self.loss = -(1-self.y).T @ np.log(1-self.y_head) - (self.y.T @ np.log(self.y_head))
        return np.sum(self.loss) / self.features

    def sigmoid(self, x): # 1 / (1+np.exp(-z))
        # This sigmoid function prevent overflowing of sigmoid function! 
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        top = np.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)
    
    def update_weights(self):
        if len(np.unique(self.y)) > 2:
            self.dw = np.zeros([self.features,self.y_onehot.shape[1]])
            self.db = 0

            self.y_head = self.forward_prop(self.X)
            self.dw = (self.X.T @ (self.y_onehot - self.y_head))  
            self.dw = self.dw / self.N
            self.db = (self.y_onehot - self.y_head).mean() 
            
            self.W -= self.dw * self.learning_rate
            self.b -= self.db * self.learning_rate
        
        if len(np.unique(self.y)) == 2:
            self.dw = np.zeros([self.features,1])
            self.db = 0

            self.y_head = self.forward_prop(self.X)
            self.dw = (self.X.T @ (self.y_onehot - self.y_head))  
            self.dw = self.dw / self.N
            self.db = np.mean(self.y_onehot - self.y_head)
            
            self.W -= self.dw * self.learning_rate
            self.b -= self.db * self.learning_rate

    def predict(self,X):
        try: 
            if len(np.unique(self.y)) > 2:
                prediction = self.forward_prop(X)
                return np.argmax(prediction,axis=1)
            if len(np.unique(self.y)) == 2:
                prediction = self.forward_prop(X)
                return np.where(prediction >= 0.5,1,0)
        except Exception: 
            return print(f"This {self.__class__.__name__} instance is not trained yet. Call 'train' with appropriate arguments before using this estimator.")


class RidgeClsf(LogisticRegression):
    def __init__(self, learning_rate=0.01,l2 = 0.01,n_iteration = 1000):
        LogisticRegression.__init__(self)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.n_iteration = n_iteration
        if type(learning_rate) != float: raise ValueError("Please enter a valid value!")
        if type(n_iteration) != int: raise ValueError("Please enter a valid value!")
        if type(l2) != float : raise ValueError("Please enter a valid value!")
    
    def update_weights(self):
        self.dw = np.ones([self.features,self.y_onehot.shape[1]])
        self.db = 0
        self.y_head = self.forward_prop(self.X)
        
        self.dw = (self.X.T @ (self.y_onehot - self.y_head)) + 2 * self.W * self.l2
        self.dw = self.dw / self.features
        self.db = (self.y_onehot - self.y_head).mean() 
        
        self.W -= self.dw * self.learning_rate
        self.b -= self.db * self.learning_rate

class LassoClsf(LogisticRegression):
    def __init__(self, learning_rate=0.01,l1 = 0.01,n_iteration = 1000):
        LogisticRegression.__init__(self)
        self.learning_rate = learning_rate
        self.l1 = l1
        self.n_iteration = n_iteration
        if type(learning_rate) != float: raise ValueError("Please enter a valid value!")
        if type(n_iteration) != int: raise ValueError("Please enter a valid value!")
        if type(l1) != float : raise ValueError("Please enter a valid value!")
    
    def update_weights(self):
        self.dw = np.ones([self.features,self.y_onehot.shape[1]])
        self.db = 0
        self.y_head = self.forward_prop(self.X)

        self.dw = (self.X.T @ (self.y_onehot - self.y_head)) + 2 * self.l1
        self.dw = self.dw / self.features
        self.db = (self.y_onehot - self.y_head).mean() 

        self.W -= self.dw * self.learning_rate
        self.b -= self.db * self.learning_rate

class ElasticNetClsf(LogisticRegression):
    def __init__(self, learning_rate=0.01,l1 = 0.01,l2=0.01, n_iteration = 1000):
        LogisticRegression.__init__(self)
        self.learning_rate = learning_rate
        self.l1 = l1
        self.n_iteration = n_iteration
        self.l2 = l2
        if type(learning_rate) != float: raise ValueError("Please enter a valid value!")
        if type(n_iteration) != int: raise ValueError("Please enter a valid value!")
        if type(l1) != float or type(l2) != float: raise ValueError("Please enter a valid value!")
        
    def update_weights(self):
        self.dw = np.ones([self.features,self.y_onehot.shape[1]])
        self.db = 0
        self.y_head = self.forward_prop(self.X)

        self.dw = (self.X.T @ (self.y_onehot - self.y_head)) + (2 * self.l1) + (2 * self.W * self.l2)
        self.dw = self.dw / self.features
        self.db = (self.y_onehot - self.y_head).mean()

        self.W -= self.dw * self.learning_rate
        self.b -= self.db * self.learning_rate

class KNN: # Look again
    def __init__(self,n_cluster,iteration):
        if type(n_cluster) != int: raise ValueError("Please enter a valid value!")
        if type(iteration) != int: raise ValueError("Please enter a valid value!")
        self.n_clusters = n_cluster
        self.iteration = iteration
        self.points = []
    
    def train(self,X: np.array):
        self.centroids = self.initialize_centroids(X)

        for i in range(self.iteration):
            old_centroids = self.centroids
            distance = self.calculate_distance(X,old_centroids)
            self.labels = self.closest_centroid(distance)
            self.centroids = self.calculate_centroids(X,self.labels)

    def transform(self,X: np.array):
        distance = self.calculate_distance(X,self.centroids)

        return self.closest_centroid(distance)
    
    def initialize_centroids(self,X):
        random_idx = centroids = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]

        return centroids

    def calculate_distance(self,X,centroids):
        distance = np.zeros((X.shape[0],self.n_clusters))

        for i in range(self.n_clusters):
            row_norm = np.linalg.norm(X - centroids[i,:],axis=1)
            distance[:,i] = np.square(row_norm)

        return distance
    
    def closest_centroid(self,distance):
        return np.argmin(distance,axis=1)

    def calculate_centroids(self,X,centroid_labels):
        centroids = np.zeros((self.n_clusters,X.shape[1]))

        for i in range(self.n_clusters):
            centroids[i,:] = np.mean(X[centroid_labels == i,:],axis=0)

        return centroids

class GausianNB:
    def train(self,X,y):
        if X.size == 0 or y.size == 0:
            raise ValueError("Size of array is zero!")
        if np.array(y).reshape(-1,1).shape[1] > 1:
            raise ValueError("Inputs' should be 1D.")
        if len(X) != len(y):
            raise ValueError("Shape of Input and Target different!")

        self.classes = np.unique(y)
        self.mean_array, self.variance_array = self.compute_mean_variance(X,y)
        self.prior_odds = self.calculate_prior(y).reshape(-1,1)
        print(self.mean_array)
    
    def guassian(self,X,mu,sigma):
        p1 = 1 / np.sqrt(2*np.pi) * sigma
        p2 = np.exp(-(X-mu)**2 / (2 * sigma ** 2))
        return p1 * p2 

    def predict(self,X: np.array):
        prob = np.zeros([X.shape[0],X.shape[1]])
        results = np.zeros([X.shape[0],len(self.classes)])
        for j in sorted(self.classes):
            for i in range(X.shape[1]):
                prob[:,i] = self.guassian(X[:,i],self.mean_array[i,j],self.variance_array[i,j])
            results[:,j] = np.prod(prob,axis=1) * self.prior_odds[j]
        return np.argmax(results,axis=1)
    
    def compute_mean_variance(self,X,y): 
        mean_array = np.zeros([X.shape[1],len(self.classes)])
        variance_array = np.zeros([X.shape[1],len(self.classes)])
        data = np.concatenate((X,y.reshape(-1,1)),axis = 1)
        for i in sorted(self.classes):
            mean_array[:,i] = np.mean(data[data[:,-1] == i][:,:-1],axis = 0)
            variance_array[:,i] = np.std(data[data[:,-1] == i][:,:-1],axis = 0)
        del data 
        return mean_array, variance_array
    
    def calculate_prior(self,y):
        self.classes = sorted(np.unique(y))
        self.priors = [len(y[y ==i]) / len(y) for i in np.unique(self.classes)]
        self.priors = np.array(self.priors).reshape(-1,1)
        return self.priors
            
class DecisionTreeClassifierr: # Not finished yet!
    def __init__(self,classification = False):
        self.classification = classification

    def train(self,X,y
    ,max_depth = 20,min_sample_split= None,min_information_gain = 1e-20):
        self.min_information_gain = min_information_gain
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split

        split_value, split_gain = self.get_best_splits(X,y)
        left,right = self.split(X,split_gain,split_value)
        print(split_gain)
        if len(X) > 0 and split_gain > self.min_information_gain:
            left = self.fit(left,y,max_depth = 20,min_sample_split= None,min_information_gain = 1e-20)
            right = self.fit(right,y,max_depth = 20,min_sample_split= None,min_information_gain = 1e-20)

    def predict(self):
        pass

    def gini_impurity(self,y):
        probability = np.array(np.unique(y,return_counts= True)[1] / y.shape[0]).reshape(-1,1)
        gini = 1 - np.sum(probability**2)
        return gini

    def entropy(self,y):
        probability = np.array(np.unique(y,return_counts= True)[1] / y.shape[0]).reshape(-1,1)
        entropy = np.sum(-probability * np.log2(probability + 1e-9))
        return entropy

    def information_gain(self,y,mask):
        child = sum(mask)
        parent = mask.shape[0] - child

        if self.classification:
            return self.variance(y) - (child / (child+parent) * self.variance(y[y == mask])) - (parent / (child+parent) * self.variance(y[y != mask]))
        else:
            return self.entropy(y)  -(child / (child+parent) * self.entropy(y[y == mask])) - (parent / (child+parent) * self.entropy(y[y != mask]))

    def variance(self,y):
        if len(y) > 1:
            return np.var(y)
        else:
            raise("Length of y smaller than 1")

    def find_combinations(self,y):
        y = np.unique(y)
        options = []
        for i in range(0,len(y)+1):
            for j in itertools.combinations(y,i):
                options.append(list(j))

        return options[1:-1]
    
    def max_information_split(self,X,y):

        information_gain = []
        split_value = []

        if self.classification:
            options = self.find_combinations(X)
        else:
            options = np.sort(np.unique(X),axis=None)[1:]

        for i in options:
            val_ig = self.information_gain(y,i)
            information_gain.append(val_ig)
            split_value.append(i)
        
        best_index = information_gain.index(max(information_gain))
        return split_value[best_index]

    def get_best_splits(self,X,y):
        split_gain = []
        split_value = []

        for i in range(0,X.shape[1]):
            split_gain.append(self.max_information_split(X[:,i],y))
            split_value.append(i)
        best_split_gain = max(split_gain)
        best_split_value = split_value[split_gain.index(best_split_gain)]
        return best_split_value,best_split_gain

    def split(self,X,split_value,label):
        if self.classification:
            X1 = X[X[:,label] == split_value]
            X2 = X[X[:,label] != split_value]
        if not self.classification:
            X1 = X[X[:,label] <= split_value]
            X2 = X[X[:,label] > split_value]
        
        return (X1,X2)

class KNNClassifier:
    def __init__(self,k_nearest=5,p=2):
        if type(p) == int: 
            self.p  = p 
        elif type(p) != int: 
            raise("Please enter a valid value for p")

        if type(p) == int:
            self.k_nearest = k_nearest
        elif type(p) != int:
            raise("Please enter a valid value for k_nearest")

    def train(self,X_train,X_test,y_train,y_test):
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = pd.DataFrame(y_train)
        self.y_test = pd.DataFrame(y_test)
        self.y_prediction = [] 
        
        for i in self.X_test:
            distances = []
            for j in self.X_train:
                distance = self.calculate_distance(i,j)
                distances.append(distance)
            
            df_dists = pd.DataFrame(data=distances, columns=['dist'],index=self.y_train.index)
            df_nn = df_dists.sort_values(by=['dist'], axis=0)[:self.k_nearest]
            # Create counter object to track the labels of k closest neighbors
            counter = [int(self.y_train[self.y_train.index == x].values[0]) for x in df_nn.index if x in self.y_train.index]
            # Get most common label of all the nearest neighbors
            prediction = max(set(counter), key = counter.count)
            self.y_prediction.append(prediction)

    def predict(self):
        return self.y_prediction

    def score(self):
        true_predictions  = [i for i,j in zip(self.y_train,self.y_test) if i == j]
        accuracy = len(true_predictions) / len(self.y_test)
        return accuracy

    def get_params(self):
        if type(self.p) == int or type(self.p) == float:
            if self.p == 1: distance = "Manhantan" 
            elif self.p == 2: distance = "Euclidean"
            else: distance = "Minkowski"
        else:
            raise("Please enter a valid value.")
        print(f"Parameters K_Nearest {self.k_nearest} and Distance is {distance}")

    def calculate_distance(self,X,y): # p == 1 Minkowski, p == 2 Euclidean p == 3 Manhantan
        length = len(X)
        distance = 0 
        if type(self.p) == int or type(self.p) == float:
            if self.p == 2 or self.p == 1:
                for i in range(length):
                    distance += np.abs(X[i] - y[i]) ** self.p
                
                return distance ** (1/self.p)
            else:
                for i in range(length):
                    distance += np.abs(X[i]- y[i])
                return distance
        else:
            raise("Please enter a valid value.")    

class KNNRegression:
    def __init__(self,k_nearest=2,p=2):
        if type(p) == int: 
            self.p  = p 
        elif type(p) != int: 
            raise("Please enter a valid value for p")

        if type(p) == int:
            self.k_nearest = k_nearest
        elif type(p) != int:
            raise("Please enter a valid value for k_nearest")

    def train(self,X_train,X_test,y_train,y_test):
        st = time.time()
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = pd.DataFrame(y_train)
        self.y_test = pd.DataFrame(y_test)
        self.y_prediction = [] 
        
        for i in self.X_test:
            distances = []
            for j in self.X_train:
                distance = self.calculate_distance(i,j)
                distances.append(distance)
            
            df_dists = pd.DataFrame(data=distances, columns=['dist'],index=self.y_train.index)
            df_nn = df_dists.sort_values(by=['dist'], axis=0)[:self.k_nearest]
            values = [int(self.y_train[self.y_train.index == x].values[0]) for x in df_nn.index if x in self.y_train.index]
            self.y_prediction.append(np.mean(values))
        print("Training took",time.time() - st)
    
    def predict(self):
        return self.y_prediction

    def score(self):
        r_square = ((self.y_prediction - self.y_pred)**2).sum() ## This is not rsquare
        return r_square

    def get_params(self):
        if type(self.p) == int or type(self.p) == float:
            if self.p == 1: distance = "Manhantan" 
            elif self.p == 2: distance = "Euclidean"
            else: distance = "Minkowski"
        else:
            raise("Please enter a valid value.")
        print(f"Parameters K_Nearest {self.k_nearest} and Distance is {distance}")

    def calculate_distance(self,X,y): # p == 1 Minkowski, p == 2 Euclidean
        length = len(X)
        distance = 0 
        if type(self.p) == int or type(self.p) == float:
            if self.p == 2 or self.p == 1:
                for i in range(length):
                    distance += np.abs(X[i] - y[i]) ** self.p
                
                return distance ** (1/self.p)
            else:
                for i in range(length):
                    distance += np.abs(X[i]- y[i])
                return distance
        else:
            raise("Please enter a valid value.")    

class LinearSVM:
    def __init__(self,learning_rate = 0.0001,regularization = 0.01,n_iterations = 1000):
        if type(learning_rate) == int or type(learning_rate) == float:
            self.learning_rate = learning_rate
        if type(regularization) == int or type(regularization) == float:
            self.regularization = regularization
        if type(n_iterations) == int:
            self.n_iterations = n_iterations
        else:
            raise("Please enter a valid value")

    def train(self,X,y):
        self.X = np.array(X)
        self.y = np.array(y).reshape([-1,1])
        self.N, self.features = self.X.shape
        self.b = 0
        self.W = np.random.randn(self.features,1)

        y_transformed = np.where(self.y <= 0, -1, 1)
        
        for _ in range(self.n_iterations):
            for index,value in enumerate(self.X):
                self.y_hat = self.forward_prop(value,self.W,self.b)
                condition = y_transformed[index] * self.y_hat >= 1
                if condition == True:
                    self.W -= self.learning_rate * (2 * self.regularization *  self.W)
                if condition == False:
                    self.W -= self.learning_rate * (2 * self.regularization * self.W - (value.reshape([-1,1]) @ y_transformed[index]).reshape([-1,1]))
                    self.b -= self.learning_rate * y_transformed[index]

        return self.W, self.b

    def score(self,y_train,y_test):
        true_predictions  = [i for i,j in zip(y_train,y_test) if i == j]
        accuracy = len(true_predictions) / len(y_test)
        return accuracy

    def predict(self,X):
        return np.sign((X @ self.W) - self.b)

    def forward_prop(self,X,W,b):
        return (X @ W) - b

class PolynomialRegression: # It works great until the 5th polynomial than generates NaN values
    def __init__(self,learning_rate = 0.001,degree = 2,regularization = 0.001,n_iterations = 100):
        if type(learning_rate) == int or type(learning_rate) == float:
            self.learning_rate = learning_rate
        if type(regularization) == int or type(regularization) == float:
            self.regularization = regularization
        if type(n_iterations) == int and type(degree) == int:
            self.n_iterations = n_iterations
            self.degree = degree
        else:
            raise("Please enter a valid value")

    def calculate_polynomials(self,X,degree):
        X_copy = X

        for i in range(2,degree+1):
            polynomial = X ** i
            X_copy = np.append(X_copy,polynomial,axis = 1)
        
        return X_copy

    def train(self,X,y):
        self.X = X
        self.y = y.reshape([-1,1])
        self.X = self.calculate_polynomials(self.X,self.degree)
        self.N, self.features = self.X.shape
        
        self.b  = 0
        self.W = np.random.randn(self.features,1)
        
        self.dw = np.zeros([self.features,1])
        self.db = 0
        
        for _ in range(self.n_iterations):
            self.dw = (self.X.T @ (self.calculate() - self.y )) /self.features
            self.W -= self.dw * self.learning_rate

            self.db = np.sum((self.calculate() - self.y)) /self.features
            self.b -= self.db * self.learning_rate

    def score(self,X_test,y_test): ## Some Problems check the truth of it
        y_hat = self.predict(X_test)
        u = ((y_test - y_hat)**2).sum()
        v = ((y_test - y_test.mean()) ** 2).sum()
        r_square = 1 - (u/v)
        return r_square

    def calculate(self):
        return (self.X @ self.W) + self.b

    def predict(self,X_test):
        X_test = self.calculate_polynomials(X_test,self.degree)
        return (X_test @ self.W) + self.b

class Perceptron: #Perceptron Classifier based on ----> "F. Rosenblatt. The perceptron, a perceiving and recognizing automaton Project Para. Cornell Aeronautical Laboratory, 1957."
    def __init__(self,learning_rate = 0.001,n_iterations = 100):
        if type(learning_rate) == int or type(learning_rate) == float:
            self.learning_rate = learning_rate
        if type(n_iterations) == int:
            self.n_iterations = n_iterations
        else:
            raise("Please enter a valid value")

    def train(self,X,y):
        self.X = X
        self.y = y.reshape([-1,1])
        self.N, self.features = self.X.shape
        self.errors = []

        self.b  = 0
        self.W = np.random.randn(self.features,1)
        for _ in range(self.n_iterations):
            update = self.y - self.predict(self.X)
            self.W += self.learning_rate * (self.X.T @ update)
            self.b += np.sum(update)
    
    def calculate(self,X) -> np.array:
        return (X @ self.W) + self.b

    def predict(self,X) -> np.array:
        return np.where(self.calculate(X) >= 0, 1, -1).reshape([-1,1])

class Adaline: # Adaline Classifier based on ----> "B. Widrow et al. Adaptive ”Adaline” neuron using chemical ”memistors”. Number Technical Report 1553-2. Stanford Electron. Labs., Stanford, CA, October 1960."
    def __init__(self,learning_rate = 0.001,n_iterations = 100):
        if type(learning_rate) == int or type(learning_rate) == float:
            self.learning_rate = learning_rate
        if type(n_iterations) == int:
            self.n_iterations = n_iterations
        else:
            raise("Please enter a valid value")

    def train(self,X,y):
        self.X = X
        self.y = y.reshape([-1,1])
        self.N, self.features = self.X.shape
        self.errors = []

        self.b  = 0
        self.W = np.random.randn(self.features,1)
        self.dw = np.zeros([self.features,1])
        self.db = 0
        for _ in range(self.n_iterations):
            self.dw = self.X.T @ (self.y - self.predict(self.X)) * self.learning_rate
            self.W += self.learning_rate * self.dw
            self.db = np.sum((self.y - self.predict(self.X))) * self.learning_rate
            self.b += self.db
    
    def calculate(self,X) -> np.array:
        return (X @ self.W) + self.b

    def cost(self) -> np.array:
        return (np.sum(self.predict(self.X) - self.y) ** 2) / 2

    def predict(self,X) -> np.array:
        return np.where(self.calculate(X) >= 0, 1, -1).reshape([-1,1])

class RandomForestClassifier:
    def __init__(self,n_trees = 10,min_samples_split = 2, max_depth = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def train(self,X,y):
        self.X = X
        self.y = y
        self.trees =  {}
        for i in range(self.n_trees):
            tree_clsf = DecisionTreeClassifier(min_samples_split=self.min_samples_split,max_depth=self.max_depth)
            X_sample,y_sample = self.sample(self.X,self.y,self.n_trees)
            self.trees[i] = tree_clsf.fit(X_sample,y_sample)
        print([x.score(self.X,self.y) for x in self.trees.values()])

    def predict(self,X_test) -> np.array:
        predictions = np.array([x.predict(X_test) for x in self.trees.values()])
        predictions = np.swapaxes(predictions,0,1)
        votes = [self.majority_voter(x) for x in predictions]
        return np.array(votes)

        
    def score(self,X_test,y_test) -> int:
        y_hat = self.predict(X_test)
        accuracy = np.sum(y_hat == y_test) / len(y_test)
        return accuracy

    @staticmethod
    def sample(X, y,n_trees) -> np.array:
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    @staticmethod
    def majority_voter(y_hat) -> int:
        counter = Counter(y_hat)
        return counter.most_common(1)[0][0]

class RandomForestRegressorr:
    def __init__(self,n_trees = 10,min_samples_split = 2, max_depth = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def train(self,X,y):
        self.X = X
        self.y = y
        self.trees =  {}
        for i in range(self.n_trees):
            tree_reg = DecisionTreeRegressor(min_samples_split=self.min_samples_split,max_depth=self.max_depth)
            X_sample,y_sample = self.sample(self.X,self.y,self.n_trees)
            self.trees[i] = tree_reg.fit(X_sample,y_sample)
    
    def predict(self,X_test) -> np.array:
        return np.mean(np.array([x.predict(X_test) for x in self.trees.values()]).T,axis=1).reshape([-1,1])

    #Fix This not r2 squared
    def score(self,X_test,y_test) -> int:
        y_hat = self.predict(X_test)
        accuracy = np.sum(y_hat == y_test) / len(y_test)
        return accuracy
        
    @staticmethod
    def sample(X, y,n_trees) -> np.array:
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

class GradientBoostingRegression: 
    def __init__(self,learning_rate = 0.1,n_estimators = 10, max_depth = 2):
        
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
    
    def train(self,X,y):
        
        self.F0 = y.mean()
        Fm = self.F0

        for _ in range(self.n_estimators):
            residuals = y - Fm
            tree = DecisionTreeRegressor(max_depth=self.max_depth,random_state=42)
            tree.fit(X,residuals)
            y_hat = tree.predict(X)
            Fm += self.learning_rate * y_hat
            self.trees.append(tree)

    def predict(self,X):
        Fm = self.F0
        for i in range(self.n_estimators):
            Fm += self.learning_rate * self.trees[i].predict(X)

        return Fm


class AdaBoostClassifier:
    def __init__(self,n_estimators=1000) :
        if type(n_estimators) == int:
            self.n_estimators = n_estimators
        else:
            raise("Please enter a valid value")

    def train(self,X,y):
        if X.size == 0 or y.size == 0:
            raise ValueError("Size of array is zero!")
        if np.array(y).reshape(-1,1).shape[1] > 1:
            raise ValueError("Inputs' should be 1D.")
        if len(X) != len(y):
            raise ValueError("Shape of Input and Target different!")
        
        self.X = X
        self.y = y
        self.y = np.where(y >= 1,1,-1)
        self.N, self.features = self.X.shape
        self.weights = np.ones([self.N])
        self.trees = []
        self.alpha = []
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=1) #Stump
            tree.fit(self.X,self.y,sample_weight=self.weights)
            self.trees.append(tree)
            self.y_hat = tree.predict(self.X)
            self.error = self.compute_error(self.weights,self.y_hat,self.y)
            self.alpha.append(self.error)
            
            self.weights *= np.exp(self.error * np.not_equal(self.y, self.y_hat).astype(int))
   
    def score(self,X_test,y_test):
        y_hat = self.predict(X_test)
        accuracy = np.sum(y_hat == y_test) / len(y_test)
        return accuracy

    def predict(self,X_test):
        predictions = np.zeros([len(X_test),self.n_estimators])
        for j,i in enumerate(self.trees):
            predictions[:,j] = (self.alpha[j] * i.predict(X_test))
        predictions = (self.error * predictions).sum(axis=1)
        predictions = np.sign(predictions)
        return predictions

    def compute_error(self,weights,y_hat,y):
        eps = 1e-10
        error = np.sum(weights * np.not_equal(y, y_hat)).astype(int) / np.sum(weights) 
        return 0.5 * np.log((1-error)/error) 

class AdaBoostRegression: #https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.21.5683&rep=rep1&type=pdf
    def __init__(self,n_estimators = 20) :
        if type(n_estimators) == int:
            self.n_estimators = n_estimators
        else:
            raise("Please enter a valid value")

    def train(self,X,y):
        if X.size == 0 or y.size == 0:
            raise ValueError("Size of array is zero!")
        if np.array(y).reshape(-1,1).shape[1] > 1:
            raise ValueError("Inputs' should be 1D.")
        if len(X) != len(y):
            raise ValueError("Shape of Input and Target different!")
        
        self.X = X
        self.y = y
        self.N, self.features = self.X.shape
        self.weights = np.repeat(1/self.N, self.N)
        self.trees = []
        self.fitted_values = np.empty((self.N, self.n_estimators))
        self.betas = []

        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=1,random_state=42) #Stump
            tree.fit(self.X,self.y,sample_weight=self.weights)
            self.y_hat = tree.predict(self.X)
            self.fitted_values[:,i] = self.y_hat
            self.trees.append(tree)
            error = np.abs(self.y-self.y_hat)
            D_ts  = np.max(error) 
            L_ts = error / D_ts
            mean_error = np.sum(self.weights * L_ts)
            if mean_error >= 0.5:
                self.T = i - 1
                self.fitted_values = self.fitted_values[:,:i-1]
                self.trees = self.trees[:i-1]
                break
            
            beta_t = mean_error / (1 - mean_error)
            Z_t = np.sum(self.weights * beta_t ** (1-L_ts))
            self.betas.append(beta_t)
            self.weights *=  beta_t**(1-L_ts)/Z_t
    
        self.model_weights = np.log(1/np.array(self.betas))
        self.y_train_hat = np.array([self.weighted_median(self.fitted_values[n], self.model_weights) for n in range(self.N)])

    def score(self,X_test,y_test):
        prediction = self.predict(X_test)
        u = ((y_test - prediction)**2).sum()
        v = ((y_test - y_test.mean())**2).sum()
        return 1 - u/v

    def predict(self, X_test):
        N_test = len(X_test)
        fitted_values = np.empty((N_test, self.n_estimators))
        for t, tree in enumerate(self.trees):
            fitted_values[:,t] = tree.predict(X_test)
        return np.array([self.weighted_median(fitted_values[n], self.model_weights) for n in range(N_test)]) 

    def weighted_median(self,values, weights):
        sorted_indices = values.argsort()
        values = values[sorted_indices]
        weights = weights[sorted_indices]
        weights_cumulative_sum = weights.cumsum()
        median_weight = np.argmax(weights_cumulative_sum >= sum(weights)/2)
        return values[median_weight]

class VotingCls:
    def __init__(self,estimators = None,voting="hard"):
        if len(estimators) == 0:
            raise AttributeError("Plase enter an estimator")
        self.estimators = estimators
        self.voting = voting
    
    def train(self,X,y):
        self.X = X
        self.y = y
        for j,i in enumerate(self.estimators):
            i.fit(X,y)
            self.estimators[j] = i
    
    def predict(self,X_test):
        predictions = np.zeros([X_test.shape[0],len(self.estimators)])
        prediction = np.zeros([X_test.shape[0],1])
        
        for j,i in enumerate(self.estimators):
            predictions[:,j] = i.predict(X_test)
        
        predictions = predictions.astype(np.int64)
        if self.voting == "hard":
            self.prediction = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
            return self.prediction
        
        if self.voting == "soft":
            pass

    def score(self,X_test,y_test):
        y_hat = self.predict(X_test)
        accuracy = np.sum(y_test == y_hat) / len(y_test)
        return accuracy

class VotingRegression:
    def __init__(self):
        pass
    
    def train(self,X,y):
        self.X = X
        self.y = y
        for j,i in enumerate(self.estimators):
            i.fit(X,y)
            self.estimators[j] = i
    
    def predict(self,X_test):
        predictions = np.zeros([X_test.shape[0],len(self.estimators)])
        prediction = np.zeros([X_test.shape[0],1])
        
        for j,i in enumerate(self.estimators):
            predictions[:,j] = i.predict(X_test)
        self.prediction = predictions.mean(axis=1)
        return prediction

    def score(self):
        u = ((self.y - self.prediction)**2).sum()
        v = ((self.y - self.y.mean())**2).sum()
        return 1 - u/v

class LDA: 
    def __init__(self,n_components = 3):
        if type(n_components) != int:
            raise ValueError("Enter a valid value for n_components.")
        self.n_components = n_components

    def fit(self,X,y):
        if X.shape[1] <= self.n_components:
            raise ValueError("Dimensions of LDA should be less than data.")
        self.X = X
        self.y = y
        self.N, self.features = self.X.shape
        self.class_labels = np.unique(y)
        
        Sw = np.zeros([self.features,self.features])
        Sb = np.zeros([self.features,self.features]) # Within class scatter matrix 
        overall_mean = np.mean(self.X,axis=0)
        
        for i in self.class_labels:
            Ci = X[y == i]
            mean_class = np.mean(Ci,axis = 0)
            Sw += (Ci - mean_class).T @ ((Ci - mean_class))

            n = Ci.shape[0]
            md = (mean_class - overall_mean).reshape([self.features,1])
            Sb += n* (md.dot(md.T))

        eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

        eigen_vectors = eigen_vectors.T

        index = np.argsort(abs(eigen_values))[::-1]
        self.eigen_values = eigen_values[index]
        eigen_vectors = eigen_vectors[index]

        self.linear_discrimants = eigen_vectors[: self.n_components]
        
    def transform(self):
        return np.dot(self.X,self.linear_discrimants.T)

    def fit_transform(self,X,y):
        self.fit(X,y)
        X_transformed = self.transform()
        return X_transformed

    def explained_variance(self):
        self.explained_var = []
        for i in range(len(self.eigen_values)):
            self.explained_var.append(np.round(self.eigen_values[i] / np.sum(self.eigen_values),2))    
        return self.explained_var

    def explained_variance_ratio(self):
        self.explained_var_ratio = []
        value = 0 
        for i in range(len(self.eigen_values)):
            self.explained_var_ratio.append(value + np.round(self.eigen_values[i] / np.sum(self.eigen_values),3))
            value += np.round(self.eigen_values[i] / np.sum(self.eigen_values),3)   
        return self.explained_var_ratio

    def visualize(self,X,y):
        X_transformed = self.fit_transform(X,y)
        plt.figure(figsize=(10,5))
        if self.n_components == 2:
            axes = sns.scatterplot(X_transformed[:,0],X_transformed[:,1],hue=y)
            plt.legend()
            return axes
        if self.n_components == 3:
            cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
            axes = plt.axes(projection = "3d")
            g = axes.scatter3D(X_transformed[:,0],X_transformed[:,1],X_transformed[:,2],c = y)
            legend = axes.legend(*g.legend_elements(), loc="upper right", title="Target Name")
            axes.add_artist(legend)
            return axes
        else:
            raise ValueError("Can't plot.")

class PCA:
    def __init__(self,n_components = 2):
        if type(n_components) != int:
            raise ValueError("Enter a valid value for n_components.")
        self.n_components = n_components
        print(self.n_components)
    def fit(self,X):
        if X.shape[1] <= self.n_components:
            raise ValueError("Dimensions of PCA should be less than data.")
        self.X = X
        self.N, self.features = self.X.shape
        means = np.mean(self.X,axis=0)
        centered = (self.X - means) / np.std(self.X,axis=0)
        covariance = np.cov(centered.T)
        self.eigen_values,self.eigen_vectors = np.linalg.eig(covariance)
        self.eigen_vectors = self.eigen_vectors.T
        return self.eigen_vectors

    def explained_variance(self):
        self.explained_var = []
        for i in range(len(self.eigen_values)):
            self.explained_var.append(np.round(self.eigen_values[i] / np.sum(self.eigen_values),3))    
        return self.explained_var

    def explained_variance_ratio(self):
        self.explained_var_ratio = []
        value = 0 
        for i in range(len(self.eigen_values)):
            self.explained_var_ratio.append(value + np.round(self.eigen_values[i] / np.sum(self.eigen_values),3))
            value += np.round(self.eigen_values[i] / np.sum(self.eigen_values),3)   
        return self.explained_var_ratio
    
    def transform(self):
        try:
            X_transformed = np.zeros([self.N,self.n_components])
            for i in range(self.n_components):
                X_transformed[:,i] = self.X @ self.eigen_vectors[:,i]
            return X_transformed
        except Exception: 
            return print(f"This {self.__class__.__name__} instance is not trained yet. Call 'train' with appropriate arguments before using this estimator.")

    def fit_transform(self,X):
        self.eigen_vectors = self.fit(X)
        X_transformed = self.transform()
        return X_transformed

    def visualize(self,X,y):
        X_transformed = self.fit_transform(X)
        if self.n_components == 2:
            axes = sns.scatterplot(X_transformed[:,0],X_transformed[:,1],hue=y)
            plt.legend()
            return axes
        if self.n_components == 3:
            plt.figure(figsize=(10,5))
            cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
            axes = plt.axes(projection = "3d")
            g = axes.scatter3D(X_transformed[:,0],X_transformed[:,1],X_transformed[:,2],c = y)
            legend = axes.legend(*g.legend_elements(), loc="upper right", title="Target Name")
            axes.add_artist(legend)
            return axes
        else:
            raise ValueError("Can't plot.")
