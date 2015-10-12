# coding: utf-8



get_ipython().magic('matplotlib inline')

import matplotlib.dates as dts
from sklearn import linear_model as lm


f = open("data_compteurgeneral_second_2015_09.csv", "r")

time_plot=dts.date2num(time)
plt.plot(time_plot[0:1440], gen[0:1440])
plt.title("Generation over time on "+(str)(time[0].date()))
plt.show()


#gen_proc is processed generation data
gen_proc = []


#loading time and gen data by day

data_time = [] # of one day
data_gen = []
list_dates = []

for i in range (0,len(time_proc[0])):
    if (time_proc[0][i].date() == time_proc[0][0].date()):    
        #print(time_proc[0][0].date())
        data_time.insert(i, time_proc[0][i])
        data_gen.insert(i, gen_proc[0][i])
    if not (time_proc[0][i].date() in list_dates) :
        list_dates.append(time_proc[0][i].date())

plt.plot(data_time, data_gen)
plt.title("Generation over time on "+(str)(data_time[0].date()))
#print((list_dates))

#stock one copy of time markers
for i in range (0,len(time_proc[0])):
    if (time_proc[0][i].date() == time_proc[0][0].date()):
        #if (float(gen_proc[0][i])>0.5):    
            data_time.insert(i, time_proc[0][i])

#stock gen data, data_gen[j] is for each date in list_dates
for j in range (0,len(list_dates)):
    data_gen.insert(j,[]) #n samples
    for i in range (0,len(time_proc[0])):
        if (time_proc[0][i].date() == list_dates[j]):
            #if (float(gen_proc[0][i])>0.5):    
            #print(time_proc[0][0].date())
                data_gen[j].append(gen_proc[0][i])

print(len(data_time))
print(len(data_gen[0]))
print(len(data_gen[1]))
print(len(data_gen[2]))
print(len(data_gen[3]))
plt.plot(data_time, data_gen[0])
plt.title("Generation over time on "+(str)(list_dates[0]))
plt.show()
plt.plot(data_time, data_gen[1])
plt.title("Generation over time on "+(str)(list_dates[1]))
plt.show()


# In[64]:

# In[70]:
#configuring time puissance i

features = []
for i in range(0,4):
    print(features[i][0]) #puissance i, temps j
print(len(data_gen))




# In[80]:
# In[78]:

X_training = np.array(features).astype(np.float)
X_training = np.reshape(X_training,(len(data_gen),4))
X_training = np.reshape(X_training,(len(features[0]),len(features)))

Y_training = np.array(data_gen).astype(np.float)
Y_training = np.reshape(Y_training,(len(Y_training),1))
Y_training = np.reshape(Y_training,(len(data_gen[0]),len(data_gen)))

X_test = X_training
Y_test = Y_training

regr = lm.LinearRegression() #classifier
regr.fit(X_training,Y_training)

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
plt.plot(data_time, data_gen[0], color = 'red')
plt.plot(data_time, data_gen[1], color = 'blue')
plt.plot(data_time, data_gen[2], color = 'black')
plt.plot(data_time, data_gen[3], color = 'green')

plt.title("Generation over time on Tuesdays in September")
plt.show()

plt.plot(data_time, regr.predict(X_test), color='blue',
plt.show()
#plt.savefig('Generation and Irradiance in a day with Linear Regression.png')

# HAVEN'T GOTTEN RID OF CONSTANT
# problems : if we do a cut off from below 10% of mean, each data set starts at different time. 
# Unable to do multivar lin reg. Unless able to leave blanks / ignore zeros


# In[ ]:
