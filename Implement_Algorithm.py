
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(45)

class Linear_Regression():
  def __init__(self, fit_intercept = True):
    # Initialize relevant variables
    '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
    '''
    self.fit_intercept = fit_intercept 
    self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
    self.all_coef=pd.DataFrame([]) # Stores the thetas for every iteration (theta vectors appended) (for the iterative methods)
    pass

  def fit_sklearn_LR(self,X,y):
    # Solve the linear regression problem by calling Linear Regression
    # from sklearn, with the relevant parameters
    model = LinearRegression()
    model.fit(X,y)
    self.fit_intercept = model.intercept_
    self.coef_ = model.coef_

    pass
  

  def mse_loss(self,y):
    y_hat = self.fit_intercept + self.coef_                
    # Compute the MSE loss with the learned model
    pass

  def compute_Batch_Gradient_Decent(self, X, batch_size, y, n_iter, lr):
    # Compute the analytical gradient (in vectorized form) of the 
    # 1. unregularized mse_loss,  and 
    # 2. mse_loss with ridge regularization
    # penalty :  specifies the regularization used  , 'l2' or unregularized
    Xnew = [[1]]*len(X)
    X = np.array(X)
    Xnew = np.concatenate((Xnew,X), axis=1)
    thetas = np.ones(X.shape[1]+1)
    temp = np.ones(X.shape[1]+1)
    no_batches = X.shape[0]//batch_size
    for i in range(1,n_iter+1):
      for j in range(no_batches): 
        X_batch = Xnew[j:j+batch_size]
        y_batch = y[j:j+batch_size]
        # if penalty == 'l2':
        #   error = -2*X_batch.T.dot(y_batch-X_batch.dot(temp)) + 1*(temp.T).dot(temp)
        #   temp = temp - (0.1/len(y_batch))*error
        # else:
        error = -2*X_batch.T.dot(y_batch-X_batch.dot(temp))/X.shape[0]
        temp = temp - ((lr)/batch_size)*error

        thetas = temp
        
    self.coef_ = thetas[1:]
    self.fit_intercept = thetas[0]

    pass

   
  def Normal_Gradient_Decent(self, X, y, n_iter, lr):
    Xnew = [[1]]*len(X)
    X = np.array(X)
    y = np.array(y)
    Xnew = np.concatenate((Xnew,X), axis=1)
  #   print(Xnew.shape)
    thetas = np.ones(X.shape[1]+1)
    temp = np.ones(X.shape[1]+1)
  #   print(thetas.shape)
    for i in range(1,n_iter+1):
        h = np.array(Xnew.dot(thetas))
        error = -2*Xnew.T.dot(y-h)/X.shape[0]
      #   print(error.shape, thetas.shape)
        temp = temp - (lr*error)
        thetas = temp
    self.coef_ = thetas[1:]
    self.fit_intercept = thetas[0]

    pass

  def Stochastic_Gradient_Descent(self, X, y, epochs, lr):
    Xnew = [[1]]*len(X)
    X = np.array(X)
    y = np.array(y)
    Xnew = np.concatenate((Xnew, X), axis = 1)
    thetas = np.ones(X.shape[1]+1)
    temp = np.ones(X.shape[1]+1)
    for j in range(epochs):

        for i in range(X.shape[0]):
            # for j in range(Xnew.shape[0]):
            h = np.array(Xnew[i].dot(thetas))
            error = -2*Xnew[i].T.dot(y[i]-h)/X.shape[0]
            temp = temp - lr*error
            thetas = temp
    self.coef_ = thetas[1:]
    self.fit_intercept = thetas[0]

    pass 

  def predict(self, X):
    # Funtion to run the LinearRegression on a test data point
    return np.dot(X,self.coef_) + self.fit_intercept
    # pass
