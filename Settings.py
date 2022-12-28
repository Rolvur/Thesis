


### Model Simulation & PV ### 

Start_date = '2021-09-06 00:00'
End_date = '2021-09-12 23:59'


Demand_pattern = 'Weekly' # 'Hourly' , 'Daily' , 'Weekly'
sEfficiency = 'k' # 'k': constant OR 'pw': piecewise 

#--------------------------------------------------------------------------
### Scenarios ### 

Start_date_scen = '2021-08-23 00:00'
End_date_scen = '2021-09-19 23:59'


## Scenario Generation ## 
Type = 'combined' # single or combined # 
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




