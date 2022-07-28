import numpy as np

class mse:
    def __call__(self,y_true,y_hat):
        return np.mean(np.power(y_true - y_hat,2))

    def grad(self,y_true,y_hat):
        return 2 * (y_hat - y_true) / y_hat.size


class binary_crossentropy:
    pass

class categorical_crossentropy:
    pass