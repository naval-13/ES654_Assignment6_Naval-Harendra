import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Implement_Algorithm import *
from metrics import *
from sklearn.model_selection import train_test_split
import time
from Implement_Algorithm import *

def predict(X, coef_, fit_intercept):
    # Funtion to run the LinearRegression on a test data point
    return np.dot(X, coef_) + fit_intercept

np.random.seed(45)
N = 500
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
# print(X.shape)

LR = Linear_Regression(fit_intercept=True)
# Call Gradient Descent here
batchsize = 20
LR.compute_gradient_batch(X,batchsize,y, 1000, 0.01)
# print(LR.coef_)
# print(LR.fit_intercept)
y_hat = LR.predict(X)

print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print("---------------------------")
