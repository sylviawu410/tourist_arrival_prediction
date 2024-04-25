# this file handle the datasets by creating the dataframe and output the correlation photo

# Import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

# import csv data sets

# CPI 
CPIfile = pd.read_csv('databases/monthlyCPI.csv')
CPIdata = CPIfile.iloc[256:520, 2] 
CPIdata = pd.to_numeric(CPIdata)

# GDP
GDPfile = pd.read_csv('databases/HKGDP.csv')
GDPdata = GDPfile.iloc[4, 181:269]

# pandemic_diseases_HK(SARS and COVID-19)
pandemicFile = pd.read_csv('databases/pandemics_in_HK.csv')
pandemicData = pandemicFile.iloc[0:265, 1]

# Average Hotel Room Occupancy Rate
hotelFile = pd.read_csv("databases/hotel_room_occupancy_rate_monthly.csv")
hotelData = hotelFile.iloc[0:264,1]

# Number of visitor arrival
visitorFile = pd.read_csv('databases/number_of_visitor_arrival.csv')
visitorData = visitorFile.iloc[0:265,1]
# visitorData = pd.to_numeric(visitorData)

# check correlations between different datasets and the vistor arrival 
def correlation_CPI():
    plt.scatter(CPIdata, visitorData)
    plt.xlabel("CPI")
    plt.ylabel("Number of visitor arrival")
    plt.yscale('log') 
    plt.show()

def correlation_pandemic():
    plt.scatter(pandemicData, visitorData)
    plt.xlabel("Pandemics in HK")
    plt.ylabel("Number of visitor arrival")
    plt.yscale('log') 
    plt.show()

def correlation_hotel():
    plt.scatter(hotelData, visitorData)
    plt.xlabel("Monthly Hotel Room Occupancy Rate")
    plt.ylabel("Number of visitor arrival")
    plt.yscale('log') 
    plt.show()

# Convert the monthly datasets to seasonal data
# Convert the index of the datasets to a DatetimeIndex
CPIdata.index = pd.to_datetime(CPIdata.index)
pandemicData.index = pd.to_datetime(pandemicData.index)
hotelData.index = pd.to_datetime(hotelData.index)
visitorData.index = pd.to_datetime(visitorData.index)

CPIdata_seasonal = CPIdata.resample('Q').mean()
pandemicData_seasonal = pandemicData.resample('Q').mean()
hotelData_seasonal = hotelData.resample('Q').mean()
visitorData_seasonal = visitorData.resample('Q').mean()

print(CPIdata_seasonal)
print(pandemicData_seasonal)
print(hotelData_seasonal)
print(visitorData_seasonal)

print("\n",CPIdata_seasonal.shape)


# Check the size of the extracted columns
def size_extracted_columns():
    print(CPIdata.shape)
    print(GDPdata.shape)
    print(pandemicData.shape)
    print(hotelData.shape)
    print(visitorData.shape)

# pd.set_option('display.max_columns', None)