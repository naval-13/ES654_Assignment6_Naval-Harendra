from typing import Union
import pandas as pd
import numpy as np


# Funtion to get predicted data
def predict(X, coef_, fit_intercept):
    return np.dot(X, coef_) + fit_intercept

def accuracy(y_cap, y):
    assert y_cap.size == y.size
    # TODO: Write here
    match_t = 0
    y = np.array(y)
    y_cap = np.array(y_cap)
    for i in range(y.size):
        if(y_cap[i]==y[i]):
            match_t +=1

    return (match_t/len(y))
    pass


def precision(y_cap, y, classs):

    match = 0
    match_t = 0
    for index, val in y_cap.iteritems():
        if val == classs:
            match_t+=1
            if val == y[index]:
                match+=1
    
    if(match_t == 0):
        print("Any value of y_cap doesn't match with the class given")
    else:
        return match/match_t
    
    pass


def recall(y_cap, y,classs):
    match = 0
    match_t = 0
    for index, val in y.iteritems():
        if val == classs:
            match_t+=1
            if val == y_cap[index]:
                match+=1
    
    if(match_t == 0):
        print("Any value of y doesn't match with the class given")
    else:
        return match/match_t

    pass


def rmse(y_cap, y):
    
    return np.sqrt(np.sum(np.square(y_cap-y))/len(y))

    pass


def mae(y_cap, y):
    
    return np.sum(abs(y_cap-y))/len(y)

    pass