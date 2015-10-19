import pandas as pd
import matplotlib.pylab as plt
docCsv = input('nom de CSV: ')
axeY = input('Axe Y : ')
broken = pd.read_csv(docCsv,index_col='Date & Time', parse_dates=['Date & Time'])
broken[axeY].plot()
plt.show()

