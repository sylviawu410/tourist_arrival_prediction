# this file handle the datasets by creating the dataframe and output the correlation figures
import pandas as pd
import matplotlib.pyplot as plt

# import csv data sets
# CPI 
CPIfile = pd.read_csv('databases/monthlyCPI.csv')
CPIdata = CPIfile.iloc[256:520, 2] 
CPIdata = pd.to_numeric(CPIdata)

# GDP (quarterly)
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

# Business Receipts Indices for tourism, convention and exhibition services
BRIfile = pd.read_csv('databases/business_receipts_indices_for_tourism.csv')
BRIdata = BRIfile.iloc[5:93,2]

# CX stock price
CXfile = pd.read_csv("databases/CX stock price.csv")
CXdata = CXfile.iloc[0:264,5]

# Convert the monthly datasets to seasonal data
index = pd.date_range('2002-01-01', periods=264, freq='M')

CPIdata.index = index
pandemicData.index = index
hotelData.index = index
visitorData.index = index
CXdata.index = index

CPIdata_seasonal = CPIdata.resample('Q').mean()
pandemicData_seasonal = pandemicData.resample('Q').mean()
hotelData_seasonal = hotelData.resample('Q').mean()
visitorData_seasonal = visitorData.resample('Q').mean()
CXdata_seasonal = CXdata.resample('Q').mean()

# datasets that are quarterly
indexQuarter = pd.date_range('2002-01-01', periods=88, freq='Q')

GDPdata_seasonal = pd.DataFrame(GDPdata_seasonal, columns=['GDP'])
GDPdata_seasonal = GDPdata_seasonal.set_index(indexQuarter)

BRIdata_seasonal = pd.DataFrame(BRIdata.values[:len(indexQuarter)], columns=['BRI'])
BRIdata_seasonal = BRIdata_seasonal.set_index(indexQuarter)

def create_df():
    combined_data = pd.concat([CPIdata_seasonal.rename("CPI"), pandemicData_seasonal, hotelData_seasonal.rename("Hotel Occupancy Rate"), visitorData_seasonal.rename("Visitor Arrival Number"), GDPdata_seasonal, BRIdata_seasonal, CXdata_seasonal.rename("CX stock price")], axis=1)
    combined_data.to_csv('dataFrame.csv', index=False)

    correlation_matrix = combined_data.corr()
    visitor_arrival_corr = correlation_matrix['Visitor Arrival Number']
    print(visitor_arrival_corr)

create_df()

def check_datasets():
    print(CPIdata.shape)
    print(GDPdata_seasonal.shape)
    print(pandemicData.shape)
    print(hotelData.shape)
    print(visitorData.shape)
    print(BRIdata.shape)

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
    GDPdata_seasonal_sorted = GDPdata_seasonal.sort_values(by='GDP')
    plt.scatter(GDPdata_seasonal_sorted['GDP'], visitorData_seasonal)
    plt.xlabel("Seasonal GDP")
    plt.ylabel("Seasonal number of visitor arrival")
    plt.xscale('log') 
    plt.show()

def correlation_BRI():
    BRIdata_sorted = BRIdata.sort_values()
    plt.scatter(BRIdata_sorted, visitorData_seasonal)
    plt.xlabel("BRI")
    plt.ylabel("Number of visitor arrival")
    plt.show()

correlation_BRI()
# pd.set_option('display.max_columns', None)