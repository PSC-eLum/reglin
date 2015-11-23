# coding: utf-8

# Préalables habituels
get_ipython().magic('matplotlib inline')
import datetime
import os
os.chdir(os.path.dirname(__file__)) ;
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from sklearn import linear_model as lm
#en plus
import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

f = open("data_compteurgeneral_second_2015_09.csv", "r")

# fréquence d’échantillonnage en Hz
fe = 1000

# durée échantillon
T = 10

# Nombre de point :
N = T*fe

# Array temporel :
t = np.arange(1.,N)/fe

# fréquence du signal : Hz
f0 = 0.5

# signal temporel
sinus = np.sin(2*np.pi*f0*t)

# ajout de bruit~\Google Drive\Polytechnique\PSC\GitHub\reglin
bruit = np.random.normal(0,0.5,N-1)

sinus2 = sinus + bruit

# signal fréquentiel : on divise par la taille du vecteur pour normaliser la fft

fourier = fftpack.fft(sinus2)/np.size(sinus2)

# axe fréquentiel:

axe_f = np.arange(0.,N-1)*fe/N

# On plot

plt.figure()

plt.subplot(121)

plt.plot(t,sinus2,'-')

plt.plot(t,sinus,'r-')

plt.xlabel('axe temporel, en seconde')

plt.subplot(122)

plt.plot(axe_f,np.abs(fourier),'x-')

plt.xlabel('axe frequentiels en Hertz')

plt.show()