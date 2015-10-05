import pandas as pd
import datetime
import csv
import matplotlib.pyplot as plt

print('Loading data ...\n');

## Load Data
data = csv.reader(open('C:\Users\damien\Desktop\PSC\Data\data_minute_over_day.csv','r'),delimiter=';')
#X = data[1][1:];
#y = data[2][1:];
header = next(data) # read the header row
column_data = zip(*data)
print(header);

X = column_data[1]
for i in X:
    i = datetime.strftime(i, %X)
y = column_data[2];

#for col in data:
#    X = col[0]    
#    y = col[1]
    
#m = y.length;

#header = next(data) read the header row
#column_data = zip(*data)


# Print out some data points
print('Le graphe: \n');

 
#plot the two lines
plt.plot(X, y)
plt.savefig("example.png")



