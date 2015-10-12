# coding: utf-8

# In[18]:


import pandas as pd
import csv
import time as t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from sklearn import linear_model as lm


# In[21]:

f = open('C:/Users/damien/Documents/PSC/Data/data_minute_over_dayRegFourrier.csv', 'r')

#C:\Users\damien\Documents\PSC\Datadata_minute_over_dayRegFourrier



f0 = []; f1 = []; f2 = []; f3 = []; f4 = []; f5 = []; f6 = []; f7 = []; f8 = []; f9 = []; f10 = [];
total = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
y = [];

for row in csv.reader(f):
    n = len(total[1])
    y.append(row[12])
    for i in range(0, len(total)):
        total[i].insert(n,row[i]);
        



X_training = np.array(total).astype(np.float)
X_training = np.reshape(X_training,(len(total[0]),11))

Y_training = np.array(y).astype(np.float)
Y_training = np.reshape(Y_training,(len(y),1))

X_test = X_training
Y_test = Y_training

regr = lm.LinearRegression() #classifier
regr.fit(X_training,Y_training)
'''
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(X_test) - Y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, Y_test))
'''
# Plot outputs
plt.plot(f0, y)
plt.title("Generation over time")
print(f1)

plt.plot(f1, regr.predict(X_test), color='blue',
         linewidth=2)

plt.ylabel("Generation [kW]")
plt.xlabel("Time")
plt.title("Predictor of Generation wrt P.P.1 (Linear Regression)")
plt.show()

#plt.savefig('Generation and Irradiance in a day with Linear Regression.png')

