
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')

import datetime
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from sklearn import linear_model as lm


# In[6]:

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
#print(len(time))


# In[47]:

time_plot=dts.date2num(time)
plt.plot(time_plot[0:1440], gen[0:1440])
plt.title("Generation over time on "+(str)(time[0].date()))
plt.show()


# In[44]:

#gen_proc is processed generation data
gen_proc = []
time_proc = []

for j in range (0,7):
    time_proc.insert(j,[])
    gen_proc.insert(j, []) #a list for each day

#first sort by weekday
for i in range(0,len(gen)):
    gen_proc[time[i].weekday()-1].append(gen[i])
    time_proc[time[i].weekday()-1].append(time[i])

#print(gen_proc[0]) #is monday


# In[48]:

data_time = [] # of one day
data_gen = []
for i in range (0,len(time_proc[0])):
    if (time_proc[0][i].date() == time_proc[0][0].date()):    
        #print(time_proc[0][0].date())
        data_time.insert(i, time_proc[0][i])
        data_gen.insert(i, gen_proc[0][i])

plt.plot(data_time, data_gen)
plt.title("Generation over time on "+(str)(data_time[0].date()))
plt.show()


# In[70]:

features = []
for i in range(0,4):
    features.insert(i,[])
    for j in range(0,len(data_time)):
        features[i].insert(j, (data_time[j].minute+data_time[j].hour*60)**i)

print(features[0][0]) #puissance i, temps j
print(features[1][0])
print(features[2][0])
print(features[3][0])
print(len(data_gen))




# In[80]:

X_training = np.array(features).astype(np.float)
X_training = np.reshape(X_training,(len(data_gen),4))

Y_training = np.array(data_gen).astype(np.float)
Y_training = np.reshape(Y_training,(len(Y_training),1))

X_test = X_training
Y_test = Y_training
#X_test = np.array(pp1[-int(size/10):]).astype(np.float)
#X_test = np.reshape(X_test,(len(X_test),1))

#Y_test = np.array(gen[-int(size/10):]).astype(np.float)
#Y_test = np.reshape(Y_test,(len(Y_test),1))

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
plt.plot(data_time, data_gen)
plt.title("Generation over time on "+(str)(data_time[0].date()))
plt.show()

plt.plot(data_time, regr.predict(X_test), color='blue',
         linewidth=2)
plt.ylabel("Generation [kW]")
plt.xlabel("Time (s)")
plt.title("Predictor of Generation wrt time (Linear Regression)")

plt.show()
#plt.savefig('Generation and Irradiance in a day with Linear Regression.png')

# HAVEN'T GOTTEN RID OF CONSTANT

