import numpy as np
from scipy.stats import t
rng = np.random.default_rng()


class One_SampleTtest():
    def __new__(self,X,value):
        self.X = np.array(X)
        self.value = value
        self.std  = np.zeros([self.X.shape[1],1])
        self.sample_mean  = np.zeros([self.X.shape[1],1])
        self.N = len(X) - 1
        if self.X.shape[1] > 1:
            for i in range(self.X.shape[1]):
                self.std[i] = np.std(X[:,i])
                self.sample_mean[i] = np.mean(X[:,i])
        
        tscore = (self.sample_mean - self.value) / (self.std / np.sqrt(self.N))
        
        return print("T-test",tscore, "p-value",2*(1 - t.cdf(abs(tscore), self.N)))


class IndependentTtest():
    def __init__(self) -> None:
        pass
    def __new__(self,X1,X2,equal_var = True):
        self.X1 = X1
        self.X2 = X2
        self.N1 = len(self.X1)
        self.N2 = len(self.X2)
        if equal_var == True:
            tscore = self.StudentstTest(self)
            pvalue = 2*(1 - t.cdf(abs(tscore), self.N1))
            print("T-Score",tscore,"p-value",pvalue)
            return tscore,pvalue
        if equal_var == False:
            tscore = self.WelchstTest(self)
            pvalue = 2*(1 - t.cdf(abs(tscore), self.N1))
            print("T-Score",tscore,"p-value",pvalue)
            return tscore,pvalue
    
    def StudentstTest(self):
        pooled_std = self.pooled_variance(self)
        t = (np.mean(self.X1) - np.mean(self.X2)) / pooled_std
        return t
    
    def WelchstTest(self):  
        estimator = self.unbiased_estimator(self)
        t = (np.mean(self.X1) - np.mean(self.X2)) / estimator
        return t

    def MannWhitneyUTest(self):
        rank = np.sort(np.concatenate(np.array(self.X1),np.array(self.X2),axis=0),axis=None)

    def pooled_variance(self):
        first_part = ((self.N1-1)*np.std(self.X1)**2) + (np.std(self.X2)**2 * (self.N2-1))
        first_part = first_part / (self.N1 + self.N2 -2)
        second_part = (1/self.N1) + (1/self.N2)
        sp = np.sqrt(first_part * second_part)
        return sp
    
    def unbiased_estimator(self):
        return np.sqrt((np.std(self.X1)**2 / self.N1) + (np.std(self.X2)**2 / self.N2))