import pyomo.environ as pe
import pyomo.opt as po
from pyomo.core import *
import pandas as pd 
import numpy as np
from Opt_Constants import *
from Data_process import Start_date,End_date, P_PV_max, Demand, c_FCR, DA, c_aFRR_up, c_aFRR_down, c_mFRR_up, π_DA, DateRange, pem_setpoint, hydrogen_mass_flow
from Settings import sEfficiency

#def ReadResults(Start_date, End_date):
def ReadResults(Start_date):
#    df_results = pd.read_excel("Result_files/Model3_"+Start_date+"_"+End_date+".xlsx")    
    df_results = pd.read_excel("Result_files/Model3_"+Start_date+".xlsx")    

    list_b_FCR = df_results["bidVol_FCR"].tolist();
    list_β_FCR = df_results["bidPrice_FCR"].tolist();
    b_FCR = dict(zip(np.arange(1,len(list_b_FCR)+1),list_b_FCR));
    β_FCR = dict(zip(np.arange(1,len(list_β_FCR)+1),list_β_FCR));

    list_b_aFRR_up = df_results["bidVol_aFRR_up"].tolist();
    list_β_aFRR_up = df_results["bidPrice_aFRR_up"].tolist();
    b_aFRR_up = dict(zip(np.arange(1,len(list_b_aFRR_up)+1),list_b_aFRR_up));
    β_aFRR_up = dict(zip(np.arange(1,len(list_β_aFRR_up)+1),list_β_aFRR_up));

    list_b_aFRR_down = df_results["bidVol_aFRR_down"].tolist();
    list_β_aFRR_down = df_results["bidPrice_aFRR_down"].tolist();
    b_aFRR_down = dict(zip(np.arange(1,len(list_b_aFRR_down)+1),list_b_aFRR_down));
    β_aFRR_down = dict(zip(np.arange(1,len(list_β_aFRR_down)+1),list_β_aFRR_down));

    list_b_mFRR_up = df_results["bidVol_mFRR_up"].tolist();
    list_β_mFRR_up = df_results["bidPrice_mFRR_up"].tolist();
    b_mFRR_up = dict(zip(np.arange(1,len(list_b_mFRR_up)+1),list_b_mFRR_up));
    β_mFRR_up = dict(zip(np.arange(1,len(list_β_mFRR_up)+1),list_β_mFRR_up));

    Φ=len(df_results.columns[df_results.columns.str.startswith('c_DA')])
    π_DA = {}
    for i in range(1,Φ+1):
        π_DA[i] = df_results['pi_DA'+str(i)].iloc[1]
    
    c_DAs = {}
    for i in range(1,Φ+1):
        for t in range(1,len(df_results)+1):
            c_DAs[(i,t)] =df_results['c_DA'+str(i)].iloc[t-1]
        #c_DA[(1,1)] = df_results['c_DA'+str(1)].iloc[1]
        #c_DA_i = {}
        #c_DA[())] = print(df_results['c_DA'+str(i)])

    return Φ, π_DA, c_DAs, b_FCR, b_aFRR_up, b_aFRR_down, b_mFRR_up, β_FCR, β_aFRR_up, β_aFRR_down, β_mFRR_up;

Φ, π_DA, c_DAs, b_FCR, b_aFRR_up, b_aFRR_down, b_mFRR_up, β_FCR, β_aFRR_up, β_aFRR_down, β_mFRR_up = ReadResults(Start_date);

    
solver = po.SolverFactory('gurobi')
SolX = pe.ConcreteModel()

T = len(P_PV_max)
SolX.T = pe.RangeSet(1,T)
#SolX.Ω = pe.RangeSet(1,Ω) No longer needed as a single reserve scenario is present (real data)
SolX.Φ = pe.RangeSet(1,Φ) # Needed, as the uncertainty of the DA-market is still present and should be taken into account
SolX.T_block = pe.RangeSet(1,T,4)

#initializing parameters
SolX.P_PV_max = pe.Param(SolX.T, initialize=P_PV_max)
SolX.c_DA = pe.Param(SolX.Φ, SolX.T, initialize=c_DAs)
SolX.DA = pe.Param(SolX.T, initialize=DA) # REAL DA clearing price
SolX.m_demand = pe.Param(SolX.T, initialize = Demand)
SolX.c_FCR = pe.Param(SolX.T,initialize = c_FCR)                        #No longer scenario dependant
SolX.c_aFRR_up = pe.Param(SolX.T, initialize = c_aFRR_up)    #No longer scenario dependant
SolX.c_aFRR_down = pe.Param(SolX.T, initialize = c_aFRR_down)#No longer scenario dependant
SolX.c_mFRR_up = pe.Param(SolX.T, initialize = c_mFRR_up)    #No longer scenario dependant
SolX.π_DA = pe.Param(SolX.Φ, initialize = π_DA)

# 1D parameters
SolX.P_pem_cap = P_pem_cap 
SolX.P_pem_min = P_pem_min
SolX.P_com = P_com
SolX.P_grid_cap = P_grid_cap
SolX.k_CR = k_CR
SolX.eff = eff
SolX.r_in = r_in
SolX.r_out = r_out
SolX.k_d = k_d
SolX.S_Pu_max = S_Pu_max
SolX.S_raw_max = S_raw_max
SolX.m_H2_max = m_H2_max
SolX.ramp_pem = ramp_pem
SolX.ramp_com = ramp_com
SolX.P_PV_cap = P_PV_cap
SolX.R_FCR_max = R_FCR_max
SolX.R_FCR_min = R_FCR_min
SolX.R_aFRR_max = R_aFRR_max #max bid size
SolX.R_aFRR_min = R_aFRR_min #min bid size 1 MW
SolX.bidres_aFRR = bidres_aFRR #100kW bid resolution
SolX.R_mFRR_max = R_mFRR_max #max bid size
SolX.R_mFRR_min = R_mFRR_min #min bid size 1 MW
SolX.bidres_mFRR = bidres_mFRR #100kW bid resolution
SolX.PT = PT
SolX.CT = CT
#defining 2D variables
SolX.z_grid = pe.Var(SolX.T, domain = pe.Binary) #binary decision variable
SolX.p_import = pe.Var(SolX.T, domain=pe.NonNegativeReals)
SolX.p_export = pe.Var(SolX.T, domain=pe.NonNegativeReals)
SolX.p_PV = pe.Var(SolX.T, domain=pe.NonNegativeReals)
SolX.p_pem = pe.Var(SolX.T, domain=pe.NonNegativeReals, bounds=(0,52.5))
SolX.m_H2 = pe.Var(SolX.T, domain=pe.NonNegativeReals, bounds=(0,1100))
SolX.m_CO2 = pe.Var(SolX.T, domain=pe.NonNegativeReals)
SolX.m_H2O = pe.Var(SolX.T, domain=pe.NonNegativeReals)
SolX.m_Ri = pe.Var(SolX.T, domain=pe.NonNegativeReals)
SolX.m_Ro = pe.Var(SolX.T, domain=pe.NonNegativeReals)
SolX.m_Pu = pe.Var(SolX.T, domain=pe.NonNegativeReals)
SolX.s_raw = pe.Var(SolX.T, domain=pe.NonNegativeReals)
SolX.s_Pu = pe.Var(SolX.T, domain=pe.NonNegativeReals)

SolX.zFCR = pe.Var(SolX.T, domain = pe.Binary)
SolX.zaFRRup = pe.Var(SolX.T, domain = pe.Binary)
SolX.zaFRRdown = pe.Var(SolX.T, domain = pe.Binary) #binary decision variable
SolX.zmFRRup = pe.Var(SolX.T, domain = pe.Binary) #binary decision variable
#Delete? SolX.bx_FCR = pe.Var(SolX.T, domain = pe.NonNegativeIntegers)
#Delete? SolX.bx_aFRR_up = pe.Var(SolX.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
#Delete? SolX.bx_aFRR_down = pe.Var(SolX.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
#Delete? SolX.bx_mFRR_up = pe.Var(SolX.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution

# Bid volume - PARAMETERS INSTEAD OF VARIABLES!
SolX.b_FCR =pe.Param(SolX.T, initialize = b_FCR)
SolX.b_aFRR_up = pe.Param(SolX.T, initialize = b_aFRR_up)
SolX.b_aFRR_down = pe.Param(SolX.T, initialize = b_aFRR_down)
SolX.b_mFRR_up = pe.Param(SolX.T, initialize = b_mFRR_up)
SolX.β_FCR =pe.Param(SolX.T, initialize = β_FCR)
SolX.β_aFRR_up = pe.Param(SolX.T, initialize = β_aFRR_up)
SolX.β_aFRR_down = pe.Param(SolX.T, initialize = β_aFRR_down)
SolX.β_mFRR_up = pe.Param(SolX.T, initialize = β_mFRR_up)

#bid acceptance binaries
SolX.δ_FCR = pe.Var(SolX.T, domain = pe.Binary) #bid acceptance binary
SolX.δ_aFRR_up = pe.Var(SolX.T, domain = pe.Binary) #bid acceptance binary
SolX.δ_aFRR_down = pe.Var(SolX.T, domain = pe.Binary) #bid acceptance binary
SolX.δ_mFRR_up = pe.Var(SolX.T, domain = pe.Binary) #bid acceptance binary

# Reserves "won"
SolX.r_FCR =pe.Var(SolX.T, domain = pe.NonNegativeReals) #Defining the variable of FCR reserve capacity
SolX.r_aFRR_up = pe.Var(SolX.T, domain = pe.NonNegativeReals)
SolX.r_aFRR_down = pe.Var(SolX.T, domain = pe.NonNegativeReals)
SolX.r_mFRR_up = pe.Var(SolX.T, domain = pe.NonNegativeReals)

SolX.vOPEX = pe.Var(SolX.T, domain = pe.Reals)

#Objective---------------------------------------------------
expr = sum((-(SolX.c_FCR[t]*SolX.r_FCR[t] + SolX.c_aFRR_up[t]*SolX.r_aFRR_up[t] + SolX.c_aFRR_down[t]*SolX.r_aFRR_down[t] + SolX.c_mFRR_up[t]*SolX.r_mFRR_up[t]) + sum(π_DA[φ]*((SolX.c_DA[φ,t]+SolX.CT)*SolX.p_import[t] - (SolX.c_DA[φ,t]-SolX.PT)*SolX.p_export[t]) for φ in SolX.Φ)) for t in SolX.T)
SolX.objective = pe.Objective(sense = pe.minimize, expr=expr)
#CONSTRAINTS---------------------------------------------------
SolX.c53_c = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_c.add((SolX.p_import[t]-SolX.p_export[t]) + SolX.p_PV[t] == SolX.p_pem[t] + SolX.P_com)

SolX.c53_de = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_de.add(SolX.p_import[t] <= SolX.z_grid[t]*SolX.P_grid_cap)
    SolX.c53_de.add(SolX.p_export[t] <= (1-SolX.z_grid[t])*SolX.P_grid_cap)

SolX.c53_f = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_f.add(SolX.p_PV[t]<= SolX.P_PV_max[t])

SolX.c53_g = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_g.add(SolX.P_pem_min <= SolX.p_pem[t])
    SolX.c53_g.add(SolX.p_pem[t] <= SolX.P_pem_cap)

#may not work after the implementation of scenarios
if sEfficiency == 'pw':
  SolX.c_piecewise = Piecewise(  SolX.T,
                          SolX.m_H2,SolX.p_pem,
                        pw_pts=pem_setpoint,
                        pw_constr_type='EQ',
                        f_rule=hydrogen_mass_flow,
                        pw_repn='SOS2')
                   
if sEfficiency == 'k':
  SolX.c53_h = pe.ConstraintList()
  for t in SolX.T:
    SolX.c53_h.add(SolX.p_pem[t] == SolX.m_H2[t]/SolX.eff)

SolX.c53_i = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_i.add(SolX.m_CO2[t] == SolX.r_in*SolX.m_H2[t])

SolX.c53_j = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_j.add(SolX.m_Ri[t] == SolX.m_H2[t] + SolX.m_CO2[t])

SolX.c53_k = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_k.add(SolX.s_raw[t] <= SolX.S_raw_max)

SolX.c53_not_included1 = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_not_included1.add(SolX.m_Ro[t] == SolX.m_Pu[t] + SolX.m_H2O[t])

SolX.c53_not_included2 = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_not_included2.add(SolX.m_Pu[t] == SolX.r_out * SolX.m_H2O[t])

SolX.c53_not_included3 = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_not_included3.add(SolX.m_Pu[t] == SolX.k_d)

SolX.c53_l = pe.ConstraintList()
for t in SolX.T:
    if t >= 2:
      SolX.c53_l.add(SolX.s_raw[t] == SolX.s_raw[t-1] + SolX.m_Ri[t] - SolX.m_Ro[t])

SolX.c53_m = pe.ConstraintList()
SolX.c53_m.add(SolX.s_raw[1] == 0.5*SolX.S_raw_max + SolX.m_Ri[1] - SolX.m_Ro[1])
SolX.c53_m.add(0.5*SolX.S_raw_max == SolX.s_raw[T])

#SolX.c14_1 = pe.ConstraintList()
#for t in SolX.T:
#  SolX.c14_1.add(0 <= SolX.s_Pu[t])

SolX.c53_n = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_n.add(SolX.s_Pu[t] <= SolX.S_Pu_max)

# Pure methanol level at "time zero" is zero, therefore the level at time 1 equals the inflow in time 1
SolX.c53_o = pe.ConstraintList()
SolX.c53_o.add(SolX.s_Pu[1] == SolX.m_Pu[1])

SolX.c53_p = pe.ConstraintList()
for t in SolX.T:
    if t >= 2:
      SolX.c53_p.add(SolX.s_Pu[t] == SolX.s_Pu[t-1] + SolX.m_Pu[t] - SolX.m_demand[t])

SolX.c53_qr = pe.ConstraintList()
for t in SolX.T:
    if t >= 2:
      SolX.c53_qr.add(-SolX.ramp_pem * SolX.P_pem_cap <= SolX.p_pem[t] - SolX.p_pem[t-1])
      SolX.c53_qr.add(SolX.p_pem[t] - SolX.p_pem[t-1] <= SolX.ramp_pem * SolX.P_pem_cap)

SolX.c53_uv = pe.ConstraintList()
M_FCR = 491.53 # max value in 2020-2021
M_aFRR_up = 154.59 # max value in 2020-2021
M_aFRR_down = 136.681 # max value in 2020-2021
M_mFRR_up = 698.31 # max value in 2020-2021
for t in SolX.T:
    SolX.c53_uv.add(SolX.c_FCR[t] - SolX.β_FCR[t] <= M_FCR*SolX.δ_FCR[t])
    SolX.c53_uv.add(SolX.c_aFRR_up[t] - SolX.β_aFRR_up[t] <= M_aFRR_up*SolX.δ_aFRR_up[t])
    SolX.c53_uv.add(SolX.c_aFRR_down[t] - SolX.β_aFRR_down[t] <= M_aFRR_down*SolX.δ_aFRR_down[t])
    SolX.c53_uv.add(SolX.c_mFRR_up[t] - SolX.β_mFRR_up[t] <= M_mFRR_up*SolX.δ_mFRR_up[t])
    SolX.c53_uv.add(SolX.β_FCR[t] - SolX.c_FCR[t] <= M_FCR * (1 - SolX.δ_FCR[t]))
    SolX.c53_uv.add(SolX.β_aFRR_up[t] - SolX.c_aFRR_up[t] <= M_aFRR_up * (1 - SolX.δ_aFRR_up[t]))
    SolX.c53_uv.add(SolX.β_aFRR_down[t] - SolX.c_aFRR_down[t] <= M_aFRR_down * (1 - SolX.δ_aFRR_down[t]))
    SolX.c53_uv.add(SolX.β_mFRR_up[t] - SolX.c_mFRR_up[t] <= M_mFRR_up * (1 - SolX.δ_mFRR_up[t]))

SolX.c53_x = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_x.add(SolX.r_FCR[t] == SolX.b_FCR[t] * SolX.δ_FCR[t])
    SolX.c53_x.add(SolX.r_aFRR_up[t] == SolX.b_aFRR_up[t] * SolX.δ_aFRR_up[t])
    SolX.c53_x.add(SolX.r_aFRR_down[t] == SolX.b_aFRR_down[t] * SolX.δ_aFRR_down[t])
    SolX.c53_x.add(SolX.r_mFRR_up[t] == SolX.b_mFRR_up[t] * SolX.δ_mFRR_up[t])


# grid constraints taking reserves into account
SolX.c53_aa = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_aa.add(SolX.P_grid_cap + (SolX.p_import[t]-SolX.p_export[t])  >= SolX.r_FCR[t] + SolX.r_aFRR_up[t] + SolX.r_mFRR_up[t])

SolX.c53_ab = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_ab.add(SolX.P_grid_cap - (SolX.p_import[t]-SolX.p_export[t])  >= SolX.r_FCR[t] + SolX.r_aFRR_down[t])

SolX.c53_ac = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_ac.add(SolX.P_pem_cap - SolX.p_pem[t]  >= SolX.r_FCR[t] + SolX.r_aFRR_down[t])

SolX.c53_ad = pe.ConstraintList()
for t in SolX.T:
    SolX.c53_ad.add(SolX.p_pem[t] - SolX.P_pem_min >= SolX.r_FCR[t] + SolX.r_aFRR_up[t] + SolX.r_mFRR_up[t])

SolX.CalcvOpex = pe.ConstraintList()
for t in SolX.T:
    SolX.CalcvOpex.add(SolX.vOPEX[t] == -(SolX.c_FCR[t]*SolX.r_FCR[t] + SolX.c_aFRR_up[t]*SolX.r_aFRR_up[t] + SolX.c_aFRR_down[t]*SolX.r_aFRR_down[t] + SolX.c_mFRR_up[t]*SolX.r_mFRR_up[t]) +(SolX.DA[t]+SolX.CT)*SolX.p_import[t] - (SolX.DA[t]-SolX.PT)*SolX.p_export[t])
###############SOLVE THE MODEL########################

#model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
Xinstance = SolX.create_instance()
Xresults = solver.solve(Xinstance)
print(Xresults)


#Converting Pyomo resulst to list
P_PEM = [Xinstance.p_pem[i].value for i in range(1,T+1)]  
P_import = [Xinstance.p_import[i].value for i in range(1,T+1)]
P_export = [Xinstance.p_export[i].value for i in range(1,T+1)]
P_grid = [P_import[i] - P_export[i] for i in range(0,len(P_import)) ]
P_PV = [Xinstance.p_PV[i].value for i in range(1,T+1)]
#b_FCR = [Xinstance.b_FCR[i].value for i in range(1,T+1)]
r_FCR = [Xinstance.r_FCR[i].value for i in range(1,T+1)]
#b_mFRRup = [Xinstance.b_mFRR_up[i].value for i in range(1,T+1)]
#β_mFRRup = [Xinstance.β_mFRR_up[i].value for i in range(1,T+1)]
r_mFRR_up = [Xinstance.r_mFRR_up[i].value for i in range(1,T+1)]
#β_aFRRup = [Xinstance.β_aFRR_up[i].value for i in range(1,T+1)]
#b_aFRRup = [Xinstance.b_aFRR_up[i].value for i in range(1,T+1)]
r_aFRR_up = [Xinstance.r_aFRR_up[i].value for i in range(1,T+1)]
#b_aFRRdown = [Xinstance.b_aFRR_down[i].value for i in range(1,T+1)]
#β_aFRRdown = [Xinstance.β_aFRR_down[i].value for i in range(1,T+1)]
r_aFRR_down = [Xinstance.r_aFRR_down[i].value for i in range(1,T+1)]
z_grid = [Xinstance.z_grid[i].value for i in range(1,T+1)]
s_raw = [Xinstance.s_raw[i].value for i in range(1,T+1)]
s_pu = [Xinstance.s_Pu[i].value for i in range(1,T+1)]
m_H2 = [Xinstance.m_H2[i].value for i in range(1,T+1)]
m_CO2 = [Xinstance.m_CO2[i].value for i in range(1,T+1)]
m_ri = [Xinstance.m_Ri[i].value for i in range(1,T+1)]
#c_FCR = [Xinstance.c_FCR[i].value for i in range(1,T+1)]
#c_aFRRup = [Xinstance.c_aFRR_up[i].value for i in range(1,T+1)]
#c_aFRRdown = [Xinstance.c_aFRR_down[i].value for i in range(1,T+1)]
#c_mFRRup = [Xinstance.c_mFRR_up[i].value for i in range(1,T+1)]
#c_DA = [Xinstance.c_DA[i].value for i in range(1,T+1)]
vOPEX = [Xinstance.vOPEX[i].value for i in range(1,T+1)]
print(sum(vOPEX))
#Creating result DataFrame
df_SolX = pd.DataFrame({#Col name : Value(list)
                          'P_PEM' : P_PEM,
                          'P_import1' : P_import,
                          'P_export1' : P_export,
                          'P_grid1' : P_grid,
                          'z_grid' : z_grid,
                          'P_PV' : P_PV,
                          'b_FCR': list(b_FCR.values()),
                          'beta_FCR': list(β_FCR.values()),
                          'r_FCR' : r_FCR,
                          'c_FCR' : list(c_FCR.values()),
                          'b_mFRR_up': list(b_mFRR_up.values()),
                          'beta_mFRR_up': list(β_mFRR_up.values()),
                          'r_mFRR_up': r_mFRR_up,
                          'c_mFRR_up' : list(c_mFRR_up.values()),
                          'b_aFRR_up': list(b_aFRR_up.values()),
                          'beta_aFRR_up': list(β_aFRR_up.values()),
                          'r_aFRR_up': r_aFRR_up,
                          'c_aFRR_up' : list(c_aFRR_up.values()),
                          'b_aFRR_down': list(b_aFRR_down.values()),
                          'beta_aFRR_down': list(β_aFRR_down.values()),
                          'r_aFRR_down': r_aFRR_down,
                          'c_aFRRdown1' : list(c_aFRR_down.values()),
                          'Raw Storage1' : s_raw,
                          'Pure Storage1' : s_pu,
                          'm_Raw_In' : m_ri,
                          'DA_clearing' : list(DA.values()),
                          'vOPEX' : vOPEX
                          }, index=DateRange,
                          )
#for i in range(1,Φ+1):
#  df_SolX['c_DA'+str(i)] = c_DA[i]
#  df_SolX['pi_DA'+str(i)] = π_DA[i]
  #m_demand = 

#save to Excel 
df_SolX.to_excel("Results_V3/SolX_"+Start_date+".xlsx")