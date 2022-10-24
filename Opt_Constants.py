#______________________CONSTANTS
P_pem_cap = 52.5 # MW capacity 
P_pem_min = 0.05*P_pem_cap
P_com = 3 #MW
P_H2O = 0.5 #MW
P_grid_cap = 238 #MW
P_PV_cap = 252 # Check again
mu_pem = 0.755 #efficiency
M_H2O = 18.01528 #g/mol
dHf0_H2O = 285830 #J/mol
M_H2 = 2.016 #g/mol
M_CO2 = 44.01 #g/mol
k_CR = mu_pem*(M_H2/dHf0_H2O)*3600000 
M_CH3OH = 32.04
r_in = (1/3)*(M_CO2/M_H2)
r_out = M_CH3OH/M_H2O
D_y = 32000000 # kg methanol / year
k_d = D_y/(365*24)
r_overhead = 1/0.9
S_Pu_max = k_d*24*7*(r_overhead)
raw_storage_days = 3
S_raw_max = k_CR*P_pem_cap*raw_storage_days*24
ramp_pem = 0.1*3600
ramp_com = 0.1*60
n_H2_max = mu_pem*P_pem_cap*1000*3600/(dHf0_H2O)
m_H2_max = n_H2_max * M_H2/1000

R_FCR_max = 1