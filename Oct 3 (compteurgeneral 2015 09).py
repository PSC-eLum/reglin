
# coding: utf-8

# In[18]:

get_ipython().magic('matplotlib inline')

import datetime
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from sklearn import linear_model as lm


# In[21]:

f = open("data_compteurgeneral_second_2015_09.csv", "r")

headers = open("Headers_CompteurGeneral.txt", "w")
for row in csv.reader(f):
    for i in range(0, len(row)):
        headers.write(row[i] + '\n')
    break
headers.close()

time = []; gen = []; use = []; pp1 = []; pp2 = []; pp3 = []; ip1 = []; ip2 = []; ip3 = []; mcb = []
total = [time, gen, use, pp1, pp2, pp3, ip1, ip2, ip3, mcb]

for row in csv.reader(f):
    n = len(total[0])
    time.insert(n,datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
    for i in range(1, len(total)):
        total[i].insert(n,row[i]);
        
#print((time[0]))
#print(type(time[0]))


# In[5]:
'''
plt.scatter(gen,pp1)
plt.xlabel("Generation [kW]")
plt.ylabel("Power Phase1 [kW]")
plt.title('Generation (kW) and Power Phase 1 (kW)')
plt.show()
#plt.savefig('Generation and Power Phase 1, September 2015.png')

plt.scatter(gen,pp2)
plt.xlabel("Generation [kW]")
plt.ylabel("Power Phase2 [kW]")
plt.title('Generation (kW) and Power Phase 2 (kW)')
plt.show()
#plt.savefig('Generation and Power Phase 2, September 2015.png')

plt.scatter(gen,pp3)
plt.xlabel("Generation [kW]")
plt.ylabel("Power Phase3 [kW]")
plt.title('Generation (kW) and Power Phase 3 (kW)')
plt.show()
#plt.savefig('Generation and Power Phase 3, September 2015.png')

plt.scatter(gen,ip1)
plt.xlabel("Generation [kW]")
plt.ylabel("Intensity Phase1 [A]")
plt.title('Generation (kW) and Intensity Phase 1 (A)')
plt.show()
#plt.savefig('Generation and Intensity Phase 1, September 2015.png')

plt.scatter(gen,ip2)
plt.xlabel("Generation [kW]")
plt.ylabel("Intensity Phase2 [A]")
plt.title('Generation (kW) and Intensity Phase 2 (A)')
plt.show()
#plt.savefig('Generation and Intensity Phase 2, September 2015.png')

plt.scatter(gen,ip3)
plt.xlabel("Generation [kW]")
plt.ylabel("Intensity Phase3 [A]")
plt.title('Generation (kW) and Intensity Phase 3 (A)')
plt.show()
#plt.savefig('Generation and Intensity Phase 3, September 2015.png')

plt.scatter(gen,mcb)
plt.xlabel("Generation [kW]")
plt.ylabel("Main Circuit Breaker [kW]")
plt.title('Generation (kW) and Main Circuit Breaker [kW]')
plt.show()
#plt.savefig('Generation and Main Circuit Breaker, September 2015.png')
'''

# In[22]:

time_plot = dts.date2num(time)
plt.plot(time_plot, gen)
plt.show()

'''
plt.plot(gen)
plt.title('Generation (kW)')
plt.show()
#plt.savefig('Generation (kW), September 2015.png')

plt.plot(pp3)
plt.title('Power Phase 3 (kW)')
#plt.show()
plt.savefig('Power Phase 3 (kW), September 2015.png')

plt.plot(ip1)
plt.title('Intensity Phase 1 (A)')
#plt.show()
plt.savefig('Intensity Phase 1 (A), September 2015.png')

plt.plot(mcb)
plt.title('Main Circuit Breaker (kW)')
#plt.show()
plt.savefig('Main Circuit Breaker (kW), September 2015.png')
'''


# In[35]:

size = len(pp1)
X_training = np.array(pp1[:-int(size/10)]).astype(np.float)
X_training = np.reshape(X_training,(len(X_training),1))

Y_training = np.array(gen[:-int(size/10)]).astype(np.float)
Y_training = np.reshape(Y_training,(len(Y_training),1))

X_test = np.array(pp1[-int(size/10):]).astype(np.float)
X_test = np.reshape(X_test,(len(X_test),1))

Y_test = np.array(gen[-int(size/10):]).astype(np.float)
Y_test = np.reshape(Y_test,(len(Y_test),1))

regr = lm.LinearRegression() #classifier
regr.fit(X_training,Y_training)

#USING SAME TRAINING SET AS TEST SET

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(X_test) - Y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, Y_test))

# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.plot(X_test, regr.predict(X_test), color='blue',
         linewidth=2)
plt.ylabel("Generation [kW]")
plt.xlabel("Power Phase 1 [kW]")
plt.title("Predictor of Generation wrt P.P.1 (Linear Regression)")

plt.show()
#plt.savefig('Generation and Irradiance in a day with Linear Regression.png')


# In[4]:




# In[ ]:



