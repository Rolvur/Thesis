#______________________CONSTANTS
#from time import pthread_getcpuclockid


P_pem_cap = 52.5 # MW capacity 
P_pem_min = 0.05*P_pem_cap
P_com = 4.5 #MW
#P_H2O = 0.5 #MW
P_grid_cap = 238 #MW
P_PV_cap = 257.2 # Due to transformer dimension
mu_pem = 0.76 #efficiency

I_x = 1.5 # from silyzer reference
mu_pem_x = 0.76 # from Silyzer reference
eff_slope = -0.102283 # slope of efficiency af function of current density - in range 0-1.5A/cm2
mu_pem_0 = mu_pem_x - (eff_slope*I_x) # efficiency at zero production

M_H2O = 18.01528 #g/mol
dHf0_H2O = 285830 #J/mol
M_H2 = 2.016 #g/mol
M_CO2 = 44.01 #g/mol

mu_slope = (mu_pem_x - mu_pem_0)/(mu_pem_x*M_H2*P_pem_cap*(1000000)*3.6/dHf0_H2O)

#k_CR = mu_pem*(M_H2/dHf0_H2O)*3600000 #Constant from power[W] to Hydrogen flow
k_CR = (M_H2/dHf0_H2O)*3600000 #Constant from power[W] to Hydrogen flow
M_CH3OH = 32.04
r_in = (1/3)*(M_CO2/M_H2)
r_out = M_CH3OH/M_H2O
D_y = 32000000 # kg methanol / year
k_d = D_y/(365*24)
r_overhead = 1/0.9  #Extra capacity in storage 
S_Pu_max = k_d*24*7*(r_overhead)
raw_storage_days = 3
S_raw_max = k_CR*P_pem_cap*raw_storage_days*24
ramp_pem = 0.1*3600
ramp_com = 0.1*60 
n_H2_max = mu_pem*P_pem_cap*1000*3600/(dHf0_H2O)
m_H2_max = n_H2_max * M_H2/1000

R_FCR_max = 1
R_FCR_min = 1
R_aFRR_max = 50 #max bid size
R_aFRR_min = 1 #min bid size 1 MW
R_mFRR_min = 5
R_mFRR_max = 50
bidres_FCR = 1
bidres_aFRR = 0.1 #100kW bid resolution
bidres_mFRR = 0.1

PT = 1.37 #Producer tariff
CT = 18.4815 #Consumer tariff 

c_CO2 = 47.4/1000 #EUR/kg
c_H2O = 0.66/1000 #EUR/kg 