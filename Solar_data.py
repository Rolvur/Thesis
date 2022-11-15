import pandas as pd 
import numpy as np
import datetime
from pathlib import Path

from os import listdir
from os.path import isfile, join

#Getting names of all files in folder 
file_names = [f for f in listdir('Solar_data') if isfile(join('Solar_data', f))]



length_day = 90000

list = np.zeros(shape=(90000, 2))


df_results = pd.DataFrame(columns = ['HourDK' , 'Irradiance' ])

file_to_open = Path("Solar_data/") / file_names[0]
df_file = pd.read_csv(file_to_open)

df_file['TIMESTAMP'] = (df_file['TIMESTAMP'].astype(int))


start_time = df_file['TIMESTAMP'][1]
end_time = df_file['TIMESTAMP'].iloc[-1]

timestamp_start = datetime.datetime.fromtimestamp(start_time)

timestamp_end = datetime.datetime.fromtimestamp(end_time)






df = pd.DataFrame({'HourDK': pd.date_range(timestamp_start, timestamp_end, freq='1H', closed='left')})

df['Irradiance'] = df_file['INSOLATION_irrad1[kW/m2]'].astype(float)
df_avg = df.groupby(pd.PeriodIndex(df['HourDK'], freq='H'))['Irradiance'].mean()
df_results = pd.concat([df_results , df_avg],ignore_index=True)

















for i in names1: 
    file_to_open = Path("Solar_data/") / i
    df_file = pd.read_csv(file_to_open)
    timestamp_start = datetime.datetime.fromtimestamp(df_file['TIMESTAMP']-(60*60))
    timestamp_end = datetime.datetime.fromtimestamp(df_file['TIMESTAMP']-(60*60))

    df = pd.DataFrame({'HourDK': pd.date_range(timestamp_start, timestamp_end, freq='1H', closed='left')})

    df['Irradiance'] = df_file['INSOLATION_irrad1[kW/m2]'].astype(float)
    df_avg = df.groupby(pd.PeriodIndex(df['HourDK'], freq='H'))['Irradiance'].mean()
    df_results = pd.concat([df_results , df_avg],ignore_index=True)
    


file.columns

file_to_open = Path("Solar_data/") / i
df_resultsM1_2020 = pd.read_excel(file_to_open)

timestamp_start = datetime.datetime.fromtimestamp(1577836800-(60*60))
timestamp_end = datetime.datetime.fromtimestamp(1640995199-(60*60))

print(timestamp_start.strftime('%Y-%m-%d %H:%M:%S'))
print(timestamp_end.strftime('%Y-%m-%d %H:%M:%S'))




df = pd.DataFrame(
        {'Hours': pd.date_range(timestamp_start, timestamp_end, freq='1H', closed='left')}
     )

df













