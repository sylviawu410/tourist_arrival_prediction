# this file handle the datasets by creating the dataframe and output the correlation figures

# Import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import csv data sets
# CPI 
CPIfile = pd.read_csv('databases/monthlyCPI.csv')
CPIdata = CPIfile.iloc[256:520, 2] 
CPIdata = pd.to_numeric(CPIdata)

# GDP (the data is already seasonal)
GDPfile = pd.read_csv('databases/HKGDP.csv')
GDPdata_seasonal = GDPfile.iloc[4, 181:269]
GDPdata_seasonal = GDPfile.iloc[4, 181:269].values.reshape(-1, 1)

# pandemic_diseases_HK(SARS and COVID-19)
pandemicFile = pd.read_csv('databases/pandemics_in_HK.csv')
pandemicData = pandemicFile.iloc[0:265, 1]

# Average Hotel Room Occupancy Rate
hotelFile = pd.read_csv("databases/hotel_room_occupancy_rate_monthly.csv")
hotelData = hotelFile.iloc[0:264,1]

# Number of visitor arrival
visitorFile = pd.read_csv('databases/number_of_visitor_arrival.csv')
visitorData = visitorFile.iloc[0:265,1]


# Convert the monthly datasets to seasonal data
index = pd.date_range('2002-01-01', periods=264, freq='M')

CPIdata.index = index
pandemicData.index = index
hotelData.index = index
visitorData.index = index

CPIdata_seasonal = CPIdata.resample('Q').mean()
pandemicData_seasonal = pandemicData.resample('Q').mean()
hotelData_seasonal = hotelData.resample('Q').mean()
visitorData_seasonal = visitorData.resample('Q').mean()



def create_df(GDPdata_seasonal):
    indexGDP = pd.date_range('2002-01-01', periods=88, freq='Q')
    GDPdata_seasonal = pd.DataFrame(GDPdata_seasonal, columns=['GDP'])
    GDPdata_seasonal = GDPdata_seasonal.set_index(indexGDP)

    combined_data = pd.concat([CPIdata_seasonal.rename("CPI"), pandemicData_seasonal, hotelData_seasonal.rename("Hotel Occupancy Rate"), visitorData_seasonal.rename("Visitor Arrival Number"), GDPdata_seasonal], axis=1)
    combined_data.to_csv('dataFrame.csv', index=False)

create_df(GDPdata_seasonal)

def check_dataset_shape():
    print(CPIdata.shape)
    print(GDPdata_seasonal.shape)
    print(pandemicData.shape)
    print(hotelData.shape)
    print(visitorData.shape)

    print("\n",CPIdata_seasonal.shape)
    print(pandemicData_seasonal.shape)
    print(hotelData_seasonal.shape)
    print(visitorData_seasonal.shape)
    print(GDPdata_seasonal.shape)

    print("\n",CPIdata_seasonal)
    print(pandemicData_seasonal)
    print(hotelData_seasonal)
    print(visitorData_seasonal)
    print(GDPdata_seasonal)


# check correlations between different datasets and the visitor arrival 
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
    plt.show()


def correlation_hotel():
    plt.scatter(hotelData, visitorData)
    plt.xlabel("Monthly Hotel Room Occupancy Rate")
    plt.ylabel("Number of visitor arrival")
    plt.yscale('log') 
    plt.show()

def correlation_GDP():
    GDPdata_seasonal_series = pd.Series(GDPdata_seasonal.flatten(), index=visitorData_seasonal.index)
    plt.scatter(GDPdata_seasonal_series, visitorData_seasonal)
    plt.xlabel("Seasonal GDP")
    plt.ylabel("Seasonal number of visitor arrival")
    plt.xscale('log') 
    plt.show()

# pd.set_option('display.max_columns', None)