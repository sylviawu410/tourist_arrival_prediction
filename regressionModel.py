# draft model by Sylvia
# Import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

# import csv data sets

# CPI
CPIfile = pd.read_csv('databases/monthlyCPI.csv')
CPIdata = CPIfile.iloc[256:520, 2] # Retrieve data from a specific row and column 
print(CPIdata)

# GDP
GDPfile = pd.read_csv('databases/HKGDP.csv')
GDPdata = GDPfile.iloc[4, 141:269]


# Latest local situation of COVID-19
COVIDfile = pd.read_csv('databases/local_covid_situation.csv')
COVIDdata = COVIDfile.iloc[0:26, 1]

# Average Hotel Room Occupancy Rate
hotelFile = pd.read_csv("databases/hotel_room_occupancy_rate_monthly.csv")
hotelData = hotelFile.iloc[0:264,1]

# Number of visitor arrival
visitorFile = pd.read_csv('databases/number_of_visitor_arrival.csv')
visitorData = visitorFile.iloc[0:265,1]


# pd.set_option('display.max_columns', None)

# check correlation
plt.scatter(CPIdata, visitorData)
plt.xlabel("CPI")
plt.ylabel("Number of visitor arrival")
plt.yscale('log') 
plt.show()
