#V1 -> V2: Include reserve products in optimization
#V2 -> V3: Include Reserve market scenarios
import pyomo.environ as pe
import pyomo.opt as po
from pyomo.core import *
import pandas as pd 
import numpy as np
from Opt_Constants import *
from Data_process import Start_date,End_date, Demand, DateRange, pem_setpoint, hydrogen_mass_flow, P_PV_max
from Settings import *
from Scenario import π_r, c_FCRs, c_aFRR_ups, c_aFRR_downs, c_mFRR_ups, Ω, c_DAs, Φ, π_DA
import csv

#____________________________________________
solver = po.SolverFactory('gurobi')
model = pe.ConcreteModel()

#Defining Sets
T = len(P_PV_max)
model.T = pe.RangeSet(1,T)
model.Ω = pe.RangeSet(1,Ω)
model.Φ = pe.RangeSet(1,Φ)
model.T_block = pe.RangeSet(1,T,4)

#Initializing parameters
model.P_PV_max = pe.Param(model.T, initialize=P_PV_max)
model.c_DA = pe.Param(model.Φ, model.T, initialize=c_DAs)
model.m_demand = pe.Param(model.T, initialize = Demand)
model.c_aFRR_up = pe.Param(model.Ω, model.T, initialize = c_aFRR_ups)
model.c_aFRR_down = pe.Param(model.Ω, model.T, initialize = c_aFRR_downs)
model.c_mFRR_up = pe.Param(model.Ω, model.T, initialize = c_mFRR_ups)
model.c_FCR = pe.Param(model.Ω,model.T, initialize = c_FCRs)
model.π_r = pe.Param(model.Ω, initialize = π_r)
model.π_DA = pe.Param(model.Φ, initialize = π_DA)
model.P_pem_cap = P_pem_cap 
model.P_pem_min = P_pem_min
model.P_com = P_com
model.P_grid_cap = P_grid_cap
model.eff = eff
model.r_in = r_in
#model.r_out = r_out
model.k_d = k_d
model.m_Ro = m_Ro
model.m_Pu = m_Pu
model.S_Pu_max = S_Pu_max
model.S_raw_max = S_raw_max
model.m_H2_max = m_H2_max
model.ramp_pem = ramp_pem
model.ramp_com = ramp_com
model.P_PV_cap = P_PV_cap
model.R_FCR_max = R_FCR_max
model.R_FCR_min = R_FCR_min
model.R_aFRR_max = R_aFRR_max #max bid size
model.R_aFRR_min = R_aFRR_min #min bid size 1 MW
model.bidres_aFRR = bidres_aFRR #100kW bid resolution
model.R_mFRR_max = R_mFRR_max #max bid size
model.R_mFRR_min = R_mFRR_min #min bid size 1 MW
model.bidres_mFRR = bidres_mFRR #100kW bid resolution
model.PT = PT
model.CT = CT

#DEFINING VARIABLES
# Power variables
model.p_import = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.p_export = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.z_grid = pe.Var(model.Ω, model.T, domain = pe.Binary) #binary decision variable
model.p_PV = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.p_pem = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals, bounds=(0,52.5))
#Mass flow variables
model.m_H2 = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals, bounds=(0,1100))
model.m_CO2 = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
#model.m_H2O = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.m_Ri = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
#model.m_Ro = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
#model.m_Pu = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
#Storage level variables
model.s_raw = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.s_Pu = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
# Reserve bid variables
model.zFCR = pe.Var(model.T, domain = pe.Binary) #Defining the first binary decision variable
model.zaFRRup = pe.Var(model.T, domain = pe.Binary) #binary decision variable
model.zaFRRdown = pe.Var(model.T, domain = pe.Binary) #binary decision variable
model.zmFRRup = pe.Var(model.T, domain = pe.Binary) #binary decision variable
model.bx_FCR = pe.Var(model.T, domain = pe.NonNegativeIntegers)
model.bx_aFRR_up = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
model.bx_aFRR_down = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
model.bx_mFRR_up = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
model.b_FCR =pe.Var(model.T, domain = pe.NonNegativeReals) #Defining the variable of FCR reserve capacity
model.b_aFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals)
model.b_aFRR_down = pe.Var(model.T, domain = pe.NonNegativeReals)
model.b_mFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals)
# Bid prices
model.β_FCR = pe.Var(model.T, domain = pe.NonNegativeReals) #
model.β_aFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals) #
model.β_aFRR_down = pe.Var(model.T, domain = pe.NonNegativeReals) #
model.β_mFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals) #
#bid acceptance binaries
model.δ_FCR = pe.Var(model.Ω, model.T, domain = pe.Binary) #bid acceptance binary
model.δ_aFRR_up = pe.Var(model.Ω, model.T, domain = pe.Binary) #bid acceptance binary
model.δ_aFRR_down = pe.Var(model.Ω, model.T, domain = pe.Binary) #bid acceptance binary
model.δ_mFRR_up = pe.Var(model.Ω, model.T, domain = pe.Binary) #bid acceptance binary
# Reserves "won"
model.r_FCR =pe.Var(model.Ω, model.T, domain = pe.NonNegativeReals) #Defining the variable of FCR reserve capacity
model.r_aFRR_up = pe.Var(model.Ω, model.T, domain = pe.NonNegativeReals)
model.r_aFRR_down = pe.Var(model.Ω, model.T, domain = pe.NonNegativeReals)
model.r_mFRR_up = pe.Var(model.Ω, model.T, domain = pe.NonNegativeReals)
model.c_obj = pe.Var(model.T, domain = pe.Reals)


#Objective function - "Expected cost" for all scenarios
expr = sum(sum(model.π_r[ω]*(-(model.c_FCR[ω,t]*model.r_FCR[ω,t] + model.c_aFRR_up[ω,t]*model.r_aFRR_up[ω,t] + model.c_aFRR_down[ω,t]*model.r_aFRR_down[ω,t] + model.c_mFRR_up[ω,t]*model.r_mFRR_up[ω,t]) + sum(π_DA[φ]*((model.c_DA[φ,t]+model.CT)*model.p_import[ω,t] - (model.c_DA[φ,t]-model.PT)*model.p_export[ω,t]) for φ in model.Φ)) for ω in model.Ω) for t in model.T)
model.objective = pe.Objective(sense = pe.minimize, expr=expr)

#Power balance constraint
model.c53_b = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_b.add((model.p_import[ω,t]-model.p_export[ω,t]) + model.p_PV[ω,t] == model.p_pem[ω,t] + model.P_com)

#Power import/export limits
model.c53_cd = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_cd.add(model.p_import[ω,t] <= model.z_grid[ω,t]*model.P_grid_cap)
    model.c53_cd.add(model.p_export[ω,t] <= (1-model.z_grid[ω,t])*model.P_grid_cap)

#PV generaiton limit (capped by solar data for given hour)
model.c53_e = pe.ConstraintList()
for ω in model.Ω: 
  for t in model.T:
    model.c53_e.add(model.p_PV[ω,t]<= model.P_PV_max[t])

model.c53_f = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_f.add(model.P_pem_min <= model.p_pem[ω,t])
    model.c53_f.add(model.p_pem[ω,t] <= model.P_pem_cap)

#Piece-wise electrolyzer efficiency - may not work after the implementation of scenarios
if sEfficiency == 'pw':
  model.c_piecewise = Piecewise(model.T,
                          model.m_H2,model.p_pem,
                        pw_pts=pem_setpoint,
                        pw_constr_type='EQ',
                        f_rule=hydrogen_mass_flow,
                        pw_repn='SOS2')
                   
if sEfficiency == 'k':
  model.c53_g = pe.ConstraintList()
  for ω in model.Ω:
    for t in model.T:
      model.c53_g.add(model.p_pem[ω,t] == model.m_H2[ω,t]/model.eff)

model.c53_h = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_h.add(model.m_CO2[ω,t] == model.r_in*model.m_H2[ω,t])

model.c53_i = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_i.add(model.m_Ri[ω,t] == model.m_H2[ω,t] + model.m_CO2[ω,t])

model.c53_j = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_j.add(model.s_raw[ω,t] <= model.S_raw_max)

model.c53_k = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c53_k.add(model.s_raw[ω,t] == model.s_raw[ω,t-1] + model.m_Ri[ω,t] - model.m_Ro)

model.c53_l = pe.ConstraintList()
for ω in model.Ω:
  model.c53_l.add(model.s_raw[ω,1] == 0.5*model.S_raw_max + model.m_Ri[ω,1] - model.m_Ro)
  model.c53_l.add(0.5*model.S_raw_max == model.s_raw[ω,T])

model.c53_m = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_m.add(model.s_Pu[ω,t] <= model.S_Pu_max)

# Pure methanol level at "time zero" is zero, therefore the level at time 1 equals the inflow in time 1
model.c53_n = pe.ConstraintList()
for ω in model.Ω:
  model.c53_n.add(model.s_Pu[ω,1] == model.m_Pu)

model.c53_o = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c53_o.add(model.s_Pu[ω,t] == model.s_Pu[ω,t-1] + model.m_Pu - model.m_demand[t])

model.c53_p = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c53_p.add(-model.ramp_pem * model.P_pem_cap <= model.p_pem[ω,t] - model.p_pem[ω,t-1])

model.c53_q = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c53_q.add(model.p_pem[ω,t] - model.p_pem[ω,t-1] <= model.ramp_pem * model.P_pem_cap)

model.c53_r = pe.ConstraintList()
for t in model.T:
  model.c53_r.add(model.bx_FCR[t] >=(R_FCR_min/bidres_FCR)* model.zFCR[t])
  model.c53_r.add(model.bx_FCR[t] <=(R_FCR_max/bidres_FCR)* model.zFCR[t])
  model.c53_r.add(model.bx_aFRR_up[t] >= (model.R_aFRR_min/model.bidres_aFRR)*model.zaFRRup[t])
  model.c53_r.add(model.bx_aFRR_up[t] <= (model.R_aFRR_max/model.bidres_aFRR)*model.zaFRRup[t])
  model.c53_r.add(model.bx_aFRR_down[t] >= (model.R_aFRR_min/model.bidres_aFRR)*model.zaFRRdown[t])
  model.c53_r.add(model.bx_aFRR_down[t] <= (model.R_aFRR_max/model.bidres_aFRR)*model.zaFRRdown[t])
  model.c53_r.add(model.bx_mFRR_up[t] >= (model.R_mFRR_min/model.bidres_mFRR)*model.zmFRRup[t])
  model.c53_r.add(model.bx_mFRR_up[t] <= (model.R_mFRR_max/model.bidres_mFRR)*model.zmFRRup[t])

model.c53_s = pe.ConstraintList()
for t in model.T:
  model.c53_s.add(model.b_FCR[t] == bidres_FCR* model.bx_FCR[t])
  model.c53_s.add(model.b_aFRR_up[t] == model.bx_aFRR_up[t]*(model.bidres_aFRR))
  model.c53_s.add(model.b_aFRR_down[t] == model.bx_aFRR_down[t]*(model.bidres_aFRR))
  model.c53_s.add(model.b_mFRR_up[t] == model.bx_mFRR_up[t]*model.bidres_mFRR)

model.c53_tu = pe.ConstraintList()
M_FCR = 491.53 # max value in 2020-2021
M_aFRR_up = 154.59 # max value in 2020-2021
M_aFRR_down = 136.681 # max value in 2020-2021
M_mFRR_up = 698.31 # max value in 2020-2021
for ω in model.Ω:
  for t in model.T:
    model.c53_tu.add(model.c_FCR[ω,t] - model.β_FCR[t] <= M_FCR*model.δ_FCR[ω,t])
    model.c53_tu.add(model.c_aFRR_up[ω,t] - model.β_aFRR_up[t] <= M_aFRR_up*model.δ_aFRR_up[ω,t])
    model.c53_tu.add(model.c_aFRR_down[ω,t] - model.β_aFRR_down[t] <= M_aFRR_down*model.δ_aFRR_down[ω,t])
    model.c53_tu.add(model.c_mFRR_up[ω,t] - model.β_mFRR_up[t] <= M_mFRR_up*model.δ_mFRR_up[ω,t])
    model.c53_tu.add(model.β_FCR[t] - model.c_FCR[ω,t] <= M_FCR * (1 - model.δ_FCR[ω,t]))
    model.c53_tu.add(model.β_aFRR_up[t] - model.c_aFRR_up[ω,t] <= M_aFRR_up * (1 - model.δ_aFRR_up[ω,t]))
    model.c53_tu.add(model.β_aFRR_down[t] - model.c_aFRR_down[ω,t] <= M_aFRR_down * (1 - model.δ_aFRR_down[ω,t]))
    model.c53_tu.add(model.β_mFRR_up[t] - model.c_mFRR_up[ω,t] <= M_mFRR_up * (1 - model.δ_mFRR_up[ω,t]))

model.c53_w = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_w.add(model.r_FCR[ω,t] == model.b_FCR[t] * model.δ_FCR[ω,t])
    model.c53_w.add(model.r_aFRR_up[ω,t] == model.b_aFRR_up[t] * model.δ_aFRR_up[ω,t])
    model.c53_w.add(model.r_aFRR_down[ω,t] == model.b_aFRR_down[t] * model.δ_aFRR_down[ω,t])
    model.c53_w.add(model.r_mFRR_up[ω,t] == model.b_mFRR_up[t] * model.δ_mFRR_up[ω,t])

model.c53_x = pe.ConstraintList()
for t in model.T_block:
    model.c53_x.add(model.b_FCR[t+1] == model.b_FCR[t])
    model.c53_x.add(model.b_FCR[t+2] == model.b_FCR[t])
    model.c53_x.add(model.b_FCR[t+3] == model.b_FCR[t]) 

model.c53_y = pe.ConstraintList()
for t in model.T_block:
    model.c53_y.add(model.β_FCR[t+1] == model.β_FCR[t])
    model.c53_y.add(model.β_FCR[t+2] == model.β_FCR[t])
    model.c53_y.add(model.β_FCR[t+3] == model.β_FCR[t]) 


# grid constraints taking reserves into account
model.c53_z = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_z.add(model.P_grid_cap + (model.p_import[ω,t]-model.p_export[ω,t])  >= model.r_FCR[ω,t] + model.r_aFRR_up[ω,t] + model.r_mFRR_up[ω,t])

model.c53_aa = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_aa.add(model.P_grid_cap - (model.p_import[ω,t]-model.p_export[ω,t])  >= model.r_FCR[ω,t] + model.r_aFRR_down[ω,t])

model.c53_ab = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_ab.add(model.P_pem_cap - model.p_pem[ω,t]  >= model.r_FCR[ω,t] + model.r_aFRR_down[ω,t])

model.c53_ac = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c53_ac.add(model.p_pem[ω,t] - model.P_pem_min >= model.r_FCR[ω,t] + model.r_aFRR_up[ω,t] + model.r_mFRR_up[ω,t])

model.c53_ad = pe.ConstraintList()
for t in model.T:
  model.c53_ad.add(model.b_FCR[t]*2 + model.b_aFRR_up[t] + model.b_aFRR_down[t] + model.b_mFRR_up[t] <= model.P_pem_cap - model.P_pem_min)

model.cObj = pe.ConstraintList()
for t in model.T:
  model.cObj.add(model.c_obj[t] == sum(model.π_r[ω]*(-(model.c_FCR[ω,t]*model.r_FCR[ω,t] + model.c_aFRR_up[ω,t]*model.r_aFRR_up[ω,t] + model.c_aFRR_down[ω,t]*model.r_aFRR_down[ω,t] + model.c_mFRR_up[ω,t]*model.r_mFRR_up[ω,t]) + sum(π_DA[φ]*((model.c_DA[φ,t]+model.CT)*model.p_import[ω,t] - (model.c_DA[φ,t]-model.PT)*model.p_export[ω,t]) for φ in model.Φ)) for ω in model.Ω))




###############SOLVE THE MODEL########################

instance = model.create_instance()
results = solver.solve(instance)
print(results)

c_DA = {}
for x in range(1, Φ+1):
    c_DA[x] = [instance.c_DA[x,i] for i in range(1,T+1)]
#Converting Pyomo results to list
b_FCR = [instance.b_FCR[i].value for i in range(1,T+1)]
β_FCR = [instance.β_FCR[i].value for i in range(1,T+1)]
b_mFRRup = [instance.b_mFRR_up[i].value for i in range(1,T+1)]
β_mFRRup = [instance.β_mFRR_up[i].value for i in range(1,T+1)]
β_aFRRup = [instance.β_aFRR_up[i].value for i in range(1,T+1)]
b_aFRRup = [instance.b_aFRR_up[i].value for i in range(1,T+1)]
b_aFRRdown = [instance.b_aFRR_down[i].value for i in range(1,T+1)]
β_aFRRdown = [instance.β_aFRR_down[i].value for i in range(1,T+1)]
Obj = [instance.c_obj[i].value for i in instance.c_obj]



pi_DA = {}
pi_DA_i = {}
for x in range(1, Φ+1):
  for i in range(1, T+1):
    pi_DA_i[i] = instance.π_DA[x]  
  #pi_DA.append(list(pi_DA_i.values()))
  pi_DA[x] = (list(pi_DA_i.values()))

#Creating result DataFrame
df_results = pd.DataFrame({#Col name : Value(list)
                          'bidVol_FCR': b_FCR,
                          'bidPrice_FCR': β_FCR,
                          'bidVol_mFRR_up': b_mFRRup,
                          'bidPrice_mFRR_up': β_mFRRup,
                          'bidVol_aFRR_up': b_aFRRup,
                          'bidPrice_aFRR_up': β_aFRRup,
                          'bidVol_aFRR_down': b_aFRRdown,
                          'bidPrice_aFRR_down': β_aFRRdown,
                          'Objective': Obj
                          }, index=DateRange,
                          )
for i in range(1,Φ+1):
  df_results['c_DA'+str(i)] = c_DA[i]
  df_results['pi_DA'+str(i)] = pi_DA[i]
  

#save to Excel 
#df_results.to_excel("Result_files/V3_Bids_"+Start_date[:10]+"_"+End_date[:10]+ ".xlsx")

df_results.to_excel("Result_files/V3_Bids_"+Type+"_"+Start_date[:10]+"_"+End_date[:10]+ ".xlsx")
## Parameter file ## 

a = [('P_pem_cap', model.P_pem_cap),
    ('P_pem_min', model.P_pem_min),
    ('P_com', model.P_com),
    ('P_grid_cap', model.P_grid_cap),
    ('r_in', model.r_in),
    ('S_Pu_max', model.S_Pu_max),
    ('S_raw_max', model.S_raw_max),
    ('m_H2_max', model.m_H2_max),
    ('ramp_pem', model.ramp_pem),
    ('ramp_com', model.ramp_com),
    ('P_PV_cap', model.P_PV_cap),
    ('PT', model.PT),
    ('CT', model.CT),
    ('Demand Pattern', Demand_pattern),
    ('Efficiency Type', sEfficiency),
    ('R_FCR_max', model.R_FCR_max),
    ('R_FCR_min', model.R_FCR_min),
    ('R_aFRR_max', model.R_aFRR_max),
    ('R_aFRR_min', model.R_aFRR_min),
    ('bidres_aFRR', model.bidres_aFRR),
    ('R_mFRR_max', model.R_mFRR_max),
    ('R_mFRR_min', model.R_mFRR_min),
    ('bidres_mFRR', model.bidres_mFRR),
    ('Type of Scen Gen', Type),
    ('n_clusters', n_clusters),
    ('n_samples', n_samples),
    ('Block size', blocksize),
    ('Sample length', sample_length)
    ]


with open("Result_files/V3_Bids_Parameters_"+Type+"_"+Start_date[:10]+"_"+End_date[:10]+ ".csv", 'w', newline='') as csvfile:
    my_writer = csv.writer(csvfile,delimiter=',')
    my_writer.writerows(a)



