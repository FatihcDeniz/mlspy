import numpy as np

# Check later for validity

class tanh:
    def __call__(self,X):
        return np.tanh(X)

    def grad(self,X):
        return 1 - np.tanh(X)**2

class sigmoid:
    def __call__(self,X):
        #pos_mask = (x >= 0)
        #neg_mask = (x < 0)
        #z = np.zeros_like(x)
        #z[pos_mask] = np.exp(-x[pos_mask])
        #z[neg_mask] = np.exp(x[neg_mask])
        #top = np.ones_like(x)
        #top[neg_mask] = z[neg_mask]
        return 1 / (1 + np.exp(-X))
    
    def grad(self,X):
        return self.__call__(X) * (1 - self.__call__(X))

class relu:
    def __call__(self, X):
        return X * (X>0)
    
    def grad(self,X):
        return 1. * (X>0)

class linear:
    def __call__(self,X):
        return X

    def grad(self,X):
        return np.ones(X.shape)

class exponential:
    def __call__(self, X):
        return np.exp(X)

    def grad(self,X):
        return np.exp(X)

class elu:
    def __call__(self,X,alpha=0.01):
        self.alpha = alpha
        return np.where(X>0,X,self.alpha*(np.exp(X)-1))
    
    def grad(self,X):
        return np.where(X>0,1,self.alpha * np.exp(X))

class lrelu:
    def __call__(self,X ,alpha = 0.01):
        self.alpha = alpha
        return np.where(X>0,X,self.alpha * X)
    
    def grad(self,X):
        return np.where(X>0,1,self.alpha)

class prelu:
    def __call__(self, X,alpha=0.01):
        self.alpha = alpha
        return np.where(X>=0,X, self.alpha*X)
    
    def grad(self,X):
        return np.where(X>=0,1,self.alpha)

class softplus:
    def __call__(self,X):
        return np.log(1 + np.exp(X))

    def grad(self,X):
        return 1 / (1 + np.exp(-X))

class softmax:
    def __call__(self,X):
        return np.exp(X) / np.sum(np.exp(X),axis=1,keepdims=True)

    def grad(self,X):
        return self.__call__(X) * (1 - self.__call__(X))

class selu:
    def __call__(self, X):
        self.alpha = 1.6732632423543772848170429916717
        self.lmd = 1.0507009873554804934193349852946
        return self.lmd * np.where(X>0, X,self.alpha * np.exp(X) - self.alpha)

    def grad(self,X):
        return self.lmd - np.where(X > 0,1, self.alpha - np.exp(X))