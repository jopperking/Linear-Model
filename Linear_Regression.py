import numpy as np 
from sklearn import linear_model
from sklearn import datasets

#dataset loading

boston = datasets.load_boston() 
print(boston)

#model learning

reg = linear_model.LinearRegression()
print(reg)

reg.fit(boston.data,boston.target)
print(reg)

#model prediction

result= reg.predict([boston.data[-1]])
print(result)
print(boston.target[-1])