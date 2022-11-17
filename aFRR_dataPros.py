import datetime
import numpy as np
import pandas as pd 
from pathlib import Path

#Reading data from csv
# Finland
file_to_open = Path("Data/") / "FI_AFFR_2020.xlsx"
df_FIafrr2020_raw = pd.read_excel(file_to_open, index_col=2)
file_to_open = Path("Data/") / "FI_AFFR_2021.xlsx"
df_FIafrr2021_raw = pd.read_excel(file_to_open, index_col=2)
file_to_open = Path("Data/") / "FI_AFFR_2022.xlsx"
df_FIafrr2022_raw = pd.read_excel(file_to_open, index_col=2)

df_FIafrr_raw = pd.concat([df_FIafrr2020_raw,df_FIafrr2021_raw,df_FIafrr2022_raw])

#'End time UTC' 
#'Start time UTC+02:00' 
#'End time UTC+02:00'
#'Automatic Frequency Restoration Reserve, price, down'
#'Automatic Frequency Restoration Reserve, capacity, up'
#'Automatic Frequency Restoration Reserve, capacity, down'
#'Automatic Frequency Restoration Reserve, price, up'


#Sweden
file_to_open = Path("Data/") / "SE_AFRR_2020.csv"
df_SEafrr2020_raw = pd.read_csv(file_to_open,sep=';',decimal=',') # decimal is used as in the csv the decimal is , and should be converted to .
file_to_open = Path("Data/") / "SE_AFRR_2021.csv"
df_SEafrr2021_raw = pd.read_csv(file_to_open,sep=';',decimal=',') # decimal is used as in the csv the decimal is , and should be converted to .
file_to_open = Path("Data/") / "SE_AFRR_2022.csv"
file_to_openn = Path("Data/") / "SE_AFRR_2022.xlsx"
#df_SEafrr2022_raw = pd.read_excel(file_to_openn, index_col=2)
#df_SEafrr2022_raw = pd.read_csv(file_to_open,sep=';',decimal=',')
file_to_open = Path("Data/") / "SE_AFRR_2022_2.csv"
df_SEafrr2022_2_raw = pd.read_csv(file_to_open,sep=';',decimal=',')

#'Period'
# #'Elområde'
#'aFRR Upp Pris (EUR/MW)'
#'aFRR Upp Volym (MW)'
#'aFRR Ned Pris (EUR/MW)'
#'aFRR Ned Volym (MW)'
#'Publiceringstidpunkt' 
#'Unnamed: 7']
#Drop last (25) rows
#Sweden
df_SEafrr2020_raw.drop(df_SEafrr2020_raw.tail(25).index, inplace=True) #Dropping last row as it is a sum
df_SEafrr2021_raw.drop(df_SEafrr2021_raw.tail(25).index, inplace=True) #Dropping last row as it is a sum

#Converting time
#Sweden
df_SEafrr2020_raw['Period'] =  df_SEafrr2020_raw['Period'].apply(pd.to_datetime)
df_SEafrr2020_raw['Publiceringstidpunkt'] =  df_SEafrr2020_raw['Publiceringstidpunkt'].apply(pd.to_datetime)
df_SEafrr2021_raw['Period'] =  df_SEafrr2021_raw['Period'].apply(pd.to_datetime)
df_SEafrr2021_raw['Publiceringstidpunkt'] =  df_SEafrr2021_raw['Publiceringstidpunkt'].apply(pd.to_datetime)
#df_SEafrr2022_raw['Period'] =  df_SEafrr2022_raw['Period'].apply(pd.to_datetime)
#df_SEafrr2022_raw['Publiceringstidpunkt'] =  df_SEafrr2022_raw['Publiceringstidpunkt'].apply(pd.to_datetime)



#combine the two
df_aFRR = pd.concat([df_SEafrr2020_raw, df_SEafrr2021_raw], ignore_index=True, sort=False)

df_aFRR.to_excel('df_aFRR.xlsx')
