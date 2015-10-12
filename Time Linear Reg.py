
# coding: utf-8

# In[209]:

# IMPORT MODULES

get_ipython().magic('matplotlib inline')

import datetime
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from sklearn import linear_model as lm
from sklearn import preprocessing
import calendar


# In[210]:

# READ DATA

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


# In[211]:

# PLOT ONE DAY

time_plot=dts.date2num(time)
plt.plot(time_plot[0:1440], gen[0:1440])
plt.title("Generation over time on "+(str)(time[0].date()))
plt.show()


# In[212]:

# PROCESS BY WEEKDAY

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


# In[213]:

# LOAD DATA FOR ALL DAYS d WHERE d is day, 0 Monday to 6 Sunday

d = 4 # d is day, 0 is monday to 6 Sunday
print("Working with " + calendar.day_name[d])

data_time = [] # of one day
data_gen = []
list_dates = []

for i in range (0,len(time_proc[d])):
    if not (time_proc[d][i].date() in list_dates) :
        list_dates.append(time_proc[d][i].date())

#print((list_dates))

#stock one copy of time markers
for i in range (0,len(time_proc[d])):
    if (time_proc[d][i].date() == time_proc[d][0].date()):
        data_time.insert(i, time_proc[d][i])

#stock gen data, data_gen[j] is for each date in list_dates
for j in range (0,len(list_dates)):
    data_gen.insert(j,[]) #n samples
    for i in range (0,len(time_proc[d])):
        if (time_proc[d][i].date() == list_dates[j]):
            data_gen[j].append(gen_proc[d][i]) 

print(len(data_time))
print(len(data_gen[0]))
print(len(data_gen[1]))
print(len(data_gen[2]))
print(len(data_gen[3]))
plt.plot(data_time, data_gen[0], color = "yellow")
plt.title("Generation over time on "+(str)(list_dates[0]))
plt.show()
plt.plot(data_time, data_gen[1], color = "red")
plt.title("Generation over time on "+(str)(list_dates[1]))
plt.show()
plt.plot(data_time, data_gen[2], color = "blue")
plt.title("Generation over time on "+(str)(list_dates[2]))
plt.show()
plt.plot(data_time, data_gen[3], color = "black")
plt.title("Generation over time on "+(str)(list_dates[3]))
plt.show()


# In[214]:

# TIME PUISSANCE i TABLE

features = []
for i in range(0,5):
    features.insert(i,[])
    features[i].insert(0, (data_time[0].minute+data_time[0].hour*60)**i)
    for j in range(1,len(data_time)):
        features[i].insert(j, (data_time[j].minute+data_time[j].hour*60)**i / features[i][0])
    features[i][0]=1

#print(features[0][0:10]) #puissance i, temps j
#print(features[1][0:10])
#print(features[2][0:10])
#print(features[3][0:10])


# In[215]:

# CUTTING GENERATION GRAPH TO WHERE DATA IS SIGNIFICANT

data_gen_coup = np.array(data_gen[1:]).astype(np.float)

#find min of where data_gen first > 0.5 and max of where data_gen last >0.5 amongst data_gen data.
pos_min = []
pos_max = []
mean = []
for i in range(0,len(data_gen_coup)-1):
    mean.insert(i,sum(data_gen_coup[i])/len(data_gen_coup[i]))
    print("mean of "+str(i)+" is "+ str(mean))
for i in range(0,len(data_gen_coup)-1):
    for j in range(0,len(data_gen_coup[0])):
        if float(data_gen_coup[i][j])>(mean[i]+1)/10:
            print("min depasses at "+str(j)+" with value of "+str(data_gen_coup[i][j])) 
            pos_min.insert(i,j)
            break
for i in range(0,len(data_gen_coup)-1):
    for j in range(len(data_gen_coup[0])-1,0,-1):
        if float(data_gen_coup[i][j])>(mean[i]+1)/10:
            print("max depasses at "+str(j)+" with value of "+str(data_gen_coup[i][j])) 
            pos_max.insert(i,j)
            break

minpass = pos_min[0]
for i in range(1,len(pos_min)):
    if pos_min[i] < minpass:
        minpass = pos_min[i]
print(minpass)

maxpass = pos_max[0]
for i in range(1,len(pos_max)):
    if pos_max[i] > max:
        maxpass = pos_max[i]
print(maxpass)


# In[216]:

# LINEAR REG

features_coup = np.array(features).astype(np.float)
X_training = np.reshape(features_coup[:,minpass:maxpass],(maxpass-minpass,len(features)))
print(np.shape(X_training))

#data_gen_coup = preprocessing.scale(data_gen_coup)
#print(data_gen_coup)
Y_training = np.reshape(data_gen_coup[:,minpass:maxpass],(maxpass-minpass,len(data_gen_coup)))

X_test = X_training
Y_test = Y_training

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
plt.plot(data_time, data_gen[0], color = 'yellow')
plt.plot(data_time, data_gen[1], color = 'blue')
plt.plot(data_time, data_gen[2], color = 'black')
plt.plot(data_time, data_gen[3], color = 'green')

plt.title("Generation over time on days "+calendar.day_name[d]+" in September")
plt.show()

plt.plot(data_time[minpass:maxpass], data_gen[0][minpass:maxpass], color='yellow')
plt.plot(data_time[minpass:maxpass], data_gen[1][minpass:maxpass], color='blue')
plt.plot(data_time[minpass:maxpass], data_gen[2][minpass:maxpass], color='black')
plt.plot(data_time[minpass:maxpass], data_gen[3][minpass:maxpass], color='green')
plt.plot(data_time[minpass:maxpass], regr.predict(X_test), color='red')
plt.ylabel("Generation [kW]")
plt.xlabel("Time (s)")
plt.title("Predictor of Generation wrt time (Linear Regression)")

plt.show()
#plt.savefig('Generation and Irradiance in a day with Linear Regression.png')

# HAVEN'T GOTTEN RID OF CONSTANT
# problems : if we do a cut off from below 10% of mean, each data set starts at different time. 
# Unable to do multivar lin reg. Unless able to leave blanks / ignore zeros


# In[217]:

# LINEAR REG FOR ONE DAY

features_coup = np.array(features).astype(np.float)
X_training = np.reshape(features_coup[:,minpass:maxpass],(maxpass-minpass,len(features)))
print(np.shape(X_training))

Y_training = np.reshape(data_gen_coup[1:2,minpass:maxpass],(maxpass-minpass,1))

X_test = X_training
Y_test = Y_training

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
plt.plot(data_time[minpass:maxpass], data_gen[1][minpass:maxpass], color='yellow')
plt.plot(data_time[minpass:maxpass], regr.predict(X_test), color='red')
plt.ylabel("Generation [kW]")
plt.xlabel("Time (s)")
plt.title("Predictor of Generation wrt time (Linear Regression)")

plt.show()


