from datetime import date


### Model Simulation & PV ### 
Start_date = '2021-07-12 00:00'
End_date = '2021-07-18 23:59'

Demand_pattern = 'Weekly' # 'Hourly' , 'Daily' , 'Weekly'
sEfficiency = 'k' # 'k': constant OR 'pw': piecewise 

#--------------------------------------------------------------------------
### Scenarios ### 
Start_date_scen = '2020-01-27 00:00'
End_date_scen = '2020-02-23 23:59'

## Scenario Generation ## 
Type = 'single' # single or combined # 
n_samples = 10000 #Number of samples to be made  
blocksize = 24 # 7days = 168 hours
sample_length = blocksize*7 # sampling 52 weeks blocks 

## Scenario Reduction ## 
n_clusters = 4

# For PV # 
PV_Cluster = 'false' ## Set 'True' to cluster weeks 

n_clusters_PV = 1
blocksize_PV = 24*7
weeks = 3




