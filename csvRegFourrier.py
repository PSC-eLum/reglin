import csv
from math import cos, pi
from operator import itemgetter

#Change time aux minutes passées à partir de 00:00:00
def dateToMinute(date):
	lst = date.split()
	temps = lst[1].split(':')
	return(int(temps[0])*60 + int(temps[1]))

def cosFeatures(temps,n):
	lst = [temps]
	for i in range(n+1):
		lst.append(round(cos(2*pi*temps*i/1440),4))
	return lst

lstCSV = []

#Copie les deux premières colonnes de dossier CSV dans la liste lstCSV
csvDoc=input("Le nom de document csv : ")
n = int(input("Le nombre de features : "))
with open(csvDoc,newline='') as file:
	reader = csv.reader(file)
	for raw in reader:
		lstCSV.append(raw[0:2])

#liste output
#Elément de lstOut est une liste [t,c0,c1,...,cn,y] où y est l'énergie consommée à t 
lstOut = []
		
for i in range(len(lstCSV)-1,0,-1):
	t = dateToMinute(lstCSV[i][0])
	lst = cosFeatures(t,n)
	lst.append(round(float(lstCSV[i][1]),4))
	lstOut.append(lst)

#output CSV's name
csvOutput = csvDoc.split('.')[0] + 'RegFourrier.csv'

#CSV output
ofile  = open(csvOutput, "w" , newline='')
writer = csv.writer(ofile)
for row in lstOut:
    writer.writerow(row)
del writer
ofile.close()