import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Implement_Algorithm import *
from metrics import *
from sklearn.model_selection import train_test_split
import time
from Implement_Algorithm import *

import streamlit as st

st.title("Visualize the effect of gradient descent for various learning rate")

#  streamlit run c:\Users\jaggi\Desktop\ML\Assignment_6_Naval-Harendra\app.py


st.header("Provide Input parameters : ")

# Default Values
LearningRate = 0.002
Epochs = 50

# Change the values according to input given
LearningRate = st.slider(' 1 )      Enter the value of Learning Rate?', 0.01, 0.999)
st.write("value of learning rate :  ", LearningRate)


Epochs = st.slider(' 2 )      Enter the Number of Epochs?', 0, 1000)
st.write("Number of Epochs :  ", Epochs)

#select the algorithm
option = st.selectbox( 'Select the algorithm you want to run',
    ('batch Gradient Descent', 'Stochastic Gradient Descent', 'Gradient Descent'))

st.write('selected Algorithm :', option)



# Data
np.random.seed(45)
N = 5000
P = 1
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

# X = 3*np.random.rand(100,1)
# y = 9 + 2*X+np.random.rand(100,1)


LR = Linear_Regression(fit_intercept=True)
if(option == "batch Gradient Descent" ):
    batchsize = 20
    LR.compute_Batch_Gradient_Decent(X,batchsize,y, Epochs, LearningRate)
elif(option == 'Stochastic Gradient Descent'):
    x = 20
else :
    x = 30


# print(LR.coef_)
# print(LR.fit_intercept)

y_predicted = LR.predict(X)
st.header("Results")
st.write("---------------------------")
st.write('RMSE: ', rmse(y_predicted, y))
st.write('MAE: ', mae(y_predicted, y))
st.write("---------------------------")


# # Create a scatter plot of the testing set with the two features on the x-axis and the target variable on the y-axis
# fig, ax = plt.subplots()
# ax.scatter(X[0], X[1], c=y, alpha=0.7)

# # Add two lines to the scatter plot, one for y_true and one for y_predicted  label='y_predicted' , label='y_true'
# ax.plot(X[0], X[1], y_predicted, c='r',label='y_predicted')
# ax.plot(X[0], X[1], y_true, c='g', label='y_true' )

# # Add a legend and axis labels
# ax.legend(loc='upper right')
# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_title('Prediction V/s Ground Truth Plot')


fig, ax = plt.subplots()
ax.scatter(X, y, label='Data')
ax.plot(X, y_predicted, color='green', label='y-predicted')


ax.legend()
ax.set_xlabel('Feature')
ax.set_ylabel('Target')
ax.set_title('Prediction V/s Truth Plot')


st.pyplot(fig)



