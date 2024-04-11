# draft model by Sylvia
# Import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

# import csv data sets
# pd.set_option('display.max_columns', None)

# CPI
CPIfile = pd.read_csv('databases/monthlyCPI.csv')
CPIdata = CPIfile.iloc[136:522, 2] # Retrieve data from a specific row and column 

# GDP
GDPfile = pd.read_csv('databases/HKGDP.csv')
GDPdata = GDPfile.iloc[4, 141:269]


# Latest local situation of COVID-19
COVIDfile = pd.read_csv('databases/local_situation_COVID-19.csv')
# COVIDdata = COVIDfile.iloc[4, 141:269]



# plt.scatter(area, price)
# plt.xlim(0, 1500)
# plt.xlabel("SaleableArea (ft)")
# plt.ylabel("SalePrice (10k hkd)")
# plt.show()
