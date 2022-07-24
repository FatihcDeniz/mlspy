import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score,r2_score
from sklearn.datasets import make_regression,make_classification,make_multilabel_classification,make_blobs
from sklearn.datasets import load_iris,load_breast_cancer,load_digits
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from machine_learning import ElasticNetClsf





if __name__ in "__main__":
    data = load_breast_cancer()
    X,y = data["data"], data["target"]
    #X,y = make_regression(n_samples=1000,n_features=10)
    X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    g = ElasticNetClsf()
    g.train(X_train,y_train)
    g.predict(X_test)
    print(accuracy_score(g.predict(X_test),y_test))
    