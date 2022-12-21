


### Model Simulation & PV ### 

Start_date = '2020-01-13 00:00'
End_date = '2020-01-19 23:59'


Start_date = '2020-06-08 00:00'
End_date = '2020-06-14 23:59'



Demand_pattern = 'Weekly' # 'Hourly' , 'Daily' , 'Weekly'
sEfficiency = 'k' # 'k': constant OR 'pw': piecewise 

#--------------------------------------------------------------------------
### Scenarios ### 

Start_date_scen = '2021-01-25 00:00'
End_date_scen = '2021-02-21 23:59'

Start_date_scen = '2021-05-11 00:00'
End_date_scen = '2021-06-07 23:59'


## Scenario Generation ## 
Type = 'single' # single or combined # 
n_samples = 10000 #Number of samples to be made  
blocksize = 24 # 7days = 168 hours
sample_length = blocksize*7 # sampling 52 weeks blocks 

## Scenario Reduction ## 
n_clusters = 4

# For PV # 
PV_Cluster = 'False' ## Set 'True' to cluster weeks 

n_clusters_PV = 1
blocksize_PV = 24*7
weeks = 1




