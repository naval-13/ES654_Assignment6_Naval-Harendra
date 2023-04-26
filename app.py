import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Implement_Algorithm import *
from metrics import *
from sklearn.model_selection import train_test_split
import time
from Implement_Algorithm import *

import streamlit as st

st.title(" Visualize the effect of gradient descent ")

#  streamlit run c:\Users\jaggi\Desktop\ML\ES654_Assignment6_Naval-Harendra\app.py


st.header(" Provide Input parameters : ")

# Default Values
LearningRate = 0.002
Epochs = 50

# Change the values according to input given
LearningRate = st.slider(' 1 ) Enter the value of Learning Rate?', 0.01, 0.999)
st.write("value of learning rate :  ", LearningRate)


Epochs = st.slider(' 2 ) Enter the Number of Epochs?', 0, 1000)
st.write("Number of Epochs :  ", Epochs)

#select the algorithm
option = st.selectbox( 'Select the algorithm you want to run',
    (   'batch Gradient Descent',
        'Stochastic Gradient Descent', 
        'Gradient Descent'
    )
    )

st.write('selected Algorithm :', option)

np.random.seed(42)

N = 500
slope = 2.3
intercept = 10
X = pd.DataFrame(np.random.uniform(0, 10, N), columns = ['X'])
y = slope * X['X'] + intercept + np.random.randn(N)
y=pd.Series(y)

LR = Linear_Regression(fit_intercept=True)

if(option == "batch Gradient Descent" ):
    batchsize = 20
    LR.compute_Batch_Gradient_Decent(X,batchsize,y, Epochs, LearningRate)
elif(option == 'Stochastic Gradient Descent'):
    LR.Stochastic_Gradient_Descent(X,y,Epochs,LearningRate)
else :
    LR.Normal_Gradient_Decent(X,y,Epochs,LearningRate)
    

# print(LR.coef_)
# print(LR.fit_intercept)

y_predicted = LR.predict(X)
st.header("Results")
st.write("---------------------------")
st.write('RMSE: ', rmse(y_predicted, y))
st.write('MAE: ', mae(y_predicted, y))
st.write("---------------------------")


fig, ax = plt.subplots()
ax.scatter(X, y, label='Data')
ax.plot(X, y_predicted, color='green', label='y-predicted')


ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Target')
ax.set_title('Prediction V/s Truth Plot')


st.pyplot(fig)



