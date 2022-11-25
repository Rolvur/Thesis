#V1 -> V2: Include reserve products in optimization
#V2 -> V3: Include Reserve market scenarios
import pyomo.environ as pe
import pyomo.opt as po
from pyomo.core import *
import pandas as pd 
import numpy as np
from Opt_Constants import *
from Data_process import P_PV_max, DA, Demand, c_FCR, c_aFRR_up, c_aFRR_down, c_mFRR_up, π, c_FCRs, c_aFRR_ups, c_aFRR_downs, c_mFRR_ups, Ω, DateRange, pem_setpoint, hydrogen_mass_flow
from Settings import sEfficiency
#____________________________________________
solver = po.SolverFactory('gurobi')
model = pe.ConcreteModel()


#set t in T
T = len(P_PV_max)
model.T = pe.RangeSet(1,T)
model.Ω = pe.RangeSet(1,Ω)
model.T_block = pe.RangeSet(1,T,4)

#initializing parameters
model.P_PV_max = pe.Param(model.T, initialize=P_PV_max)
model.DA = pe.Param(model.T, initialize=DA)
model.m_demand = pe.Param(model.T, initialize = Demand)
model.c_FCR = pe.Param(model.Ω,model.T,initialize = c_FCRs)
model.c_aFRR_up = pe.Param(model.Ω, model.T, initialize = c_aFRR_ups)
model.c_aFRR_down = pe.Param(model.Ω, model.T, initialize = c_aFRR_downs)
model.c_mFRR_up = pe.Param(model.Ω, model.T, initialize = c_mFRR_ups)
model.π = pe.Param(model.Ω, initialize = π)

model.P_pem_cap = P_pem_cap 
model.P_pem_min = P_pem_min
model.P_com = P_com
model.P_grid_cap = P_grid_cap
model.k_CR = k_CR
model.eff = eff
model.r_in = r_in
model.r_out = r_out
model.k_d = k_d
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
#defining variables
#model.p_grid = pe.Var(model.T, domain=pe.Reals)
model.p_import = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.p_export = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.p_PV = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.p_pem = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals, bounds=(0,52.5))
model.m_H2 = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals, bounds=(0,1100))
model.m_CO2 = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.m_H2O = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.m_Ri = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.m_Ro = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.m_Pu = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.s_raw = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.s_Pu = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)

model.zFCR = pe.Var(model.T, domain = pe.Binary) #Defining the first binary decision variable
model.zaFRRup = pe.Var(model.T, domain = pe.Binary) #binary decision variable
model.zaFRRdown = pe.Var(model.T, domain = pe.Binary) #binary decision variable
model.zmFRRup = pe.Var(model.T, domain = pe.Binary) #binary decision variable
model.bx_FCR = pe.Var(model.T, domain = pe.NonNegativeIntegers)
model.bx_aFRR_up = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
model.bx_aFRR_down = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
model.bx_mFRR_up = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution

# Bid volume
model.b_FCR =pe.Var(model.T, domain = pe.NonNegativeReals) #Defining the variable of FCR reserve capacity
model.b_aFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals)
model.b_aFRR_down = pe.Var(model.T, domain = pe.NonNegativeReals)
model.b_mFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals)
model.z_grid = pe.Var(model.Ω, model.T, domain = pe.Binary) #binary decision variable
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


#Objective
expr = sum(sum(-model.π[ω]*((model.c_FCR[ω,t]*model.r_FCR[ω,t] + model.c_aFRR_up[ω,t]*model.r_aFRR_up[ω,t] + model.c_aFRR_down[ω,t]*model.r_aFRR_down[ω,t] + model.c_mFRR_up[ω,t]*model.r_mFRR_up[ω,t]) + (model.DA[t]+model.CT)*model.p_import[ω,t] - (model.DA[t]-model.PT)*model.p_export[ω,t]) for ω in model.Ω) for t in model.T)
model.objective = pe.Objective(sense = pe.minimize, expr=expr)

model.c_a = pe.ConstraintList()
M_FCR = 3000 #
M_aFRR_up = 3000 #
M_aFRR_down = 3000 #
M_mFRR_up = 3000 #
for ω in model.Ω:
  for t in model.T:
    model.c_a.add(model.c_FCR[ω,t] - model.β_FCR[t] <= M_FCR*model.δ_FCR[ω,t])
    model.c_a.add(model.c_aFRR_up[ω,t] - model.β_aFRR_up[t] <= M_aFRR_up*model.δ_aFRR_up[ω,t])
    model.c_a.add(model.c_aFRR_down[ω,t] - model.β_aFRR_down[t] <= M_aFRR_down*model.δ_aFRR_down[ω,t])
    model.c_a.add(model.c_mFRR_up[ω,t] - model.β_mFRR_up[t] <= M_mFRR_up*model.δ_mFRR_up[ω,t])
    model.c_a.add(model.β_FCR[t] - model.c_FCR[ω,t] <= M_FCR * (1 - model.δ_FCR[ω,t]))
    model.c_a.add(model.β_aFRR_up[t] - model.c_aFRR_up[ω,t] <= M_aFRR_up * (1 - model.δ_aFRR_up[ω,t]))
    model.c_a.add(model.β_aFRR_down[t] - model.c_aFRR_down[ω,t] <= M_aFRR_down * (1 - model.δ_aFRR_down[ω,t]))
    model.c_a.add(model.β_mFRR_up[t] - model.c_mFRR_up[ω,t] <= M_mFRR_up * (1 - model.δ_mFRR_up[ω,t]))

model.c_b = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c_b.add(model.r_FCR[ω,t] == model.b_FCR[t] * model.δ_FCR[ω,t])
    model.c_b.add(model.r_aFRR_up[ω,t] == model.b_aFRR_up[t] * model.δ_aFRR_up[ω,t])
    model.c_b.add(model.r_aFRR_down[ω,t] == model.b_aFRR_down[t] * model.δ_aFRR_down[ω,t])
    model.c_b.add(model.r_mFRR_up[ω,t] == model.b_mFRR_up[t] * model.δ_mFRR_up[ω,t])


#creating a set of constraints
model.c1 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c1.add((model.p_import[ω,t]-model.p_export[ω,t]) + model.p_PV[ω,t] == model.p_pem[ω,t] + model.P_com)

#Constraint 2.1
#model.c2_1 = pe.ConstraintList()
#for t in model.T:
#    model.c2_1.add(-model.P_grid_cap <= model.p_grid[t])

#Constraint 2.2
#model.c2_2 = pe.ConstraintList()
#for t in model.T:
#    model.c2_2.add(model.p_grid[t] <= model.P_grid_cap)

#New Constraint
model.c2 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c2.add(model.p_import[ω,t] <= model.z_grid[ω,t]*model.P_grid_cap)
    model.c2.add(model.p_export[ω,t] <= (1-model.z_grid[ω,t])*model.P_grid_cap)


#Constraint 3
model.c3 = pe.ConstraintList()
for ω in model.Ω: 
  for t in model.T:
    model.c3.add(model.p_PV[ω,t]<= model.P_PV_max[t])

model.c4_1 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c4_1.add(model.P_pem_min <= model.p_pem[ω,t])

model.c4_2 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c4_2.add(model.p_pem[ω,t] <= model.P_pem_cap)

#may not work after the implementation of scenarios
if sEfficiency == 'pw':
  model.c_piecewise = Piecewise(  model.T,
                          model.m_H2,model.p_pem,
                        pw_pts=pem_setpoint,
                        pw_constr_type='EQ',
                        f_rule=hydrogen_mass_flow,
                        pw_repn='SOS2')
                   
if sEfficiency == 'k':
  model.csimple = pe.ConstraintList()
  for ω in model.Ω:
    for t in model.T:
      model.csimple.add(model.p_pem[ω,t] == model.m_H2[ω,t]/model.eff)

model.c6 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c6.add(model.m_CO2[ω,t] == model.r_in*model.m_H2[ω,t])

model.c7 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c7.add(model.m_Ri[ω,t] == model.m_H2[ω,t] + model.m_CO2[ω,t])

model.c8 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c8.add(model.m_Ro[ω,t] == model.m_Pu[ω,t] + model.m_H2O[ω,t])

model.c9 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c9.add(model.m_Pu[ω,t] == model.r_out * model.m_H2O[ω,t])

model.c10 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c10.add(model.m_Pu[ω,t] == model.k_d)

#model.c11_1 = pe.ConstraintList()
#for t in model.T:
#    model.c11_1.add(0 <= model.s_raw[t])

model.c11_2 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c11_2.add(model.s_raw[ω,t] <= model.S_raw_max)

model.c12 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c12.add(model.s_raw[ω,t] == model.s_raw[ω,t-1] + model.m_Ri[ω,t] - model.m_Ro[ω,t])

model.c13_1 = pe.Constraint(expr=model.s_raw[ω,1] == 0.5*model.S_raw_max + model.m_Ri[ω,1] - model.m_Ro[ω,1])

model.c13_2 = pe.Constraint(expr=0.5*model.S_raw_max == model.s_raw[ω,T])

#model.c14_1 = pe.ConstraintList()
#for t in model.T:
#  model.c14_1.add(0 <= model.s_Pu[t])

model.c14_2 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c14_2.add(model.s_Pu[ω,t] <= model.S_Pu_max)

# Pure methanol level at "time zero" is zero, therefore the level at time 1 equals the inflow in time 1
model.c15 = pe.Constraint(expr = model.s_Pu[ω,1] == model.m_Pu[ω,1])

model.c16 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c16.add(model.s_Pu[ω,t] == model.s_Pu[ω,t-1] + model.m_Pu[ω,t] - model.m_demand[t])

model.c17_1 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c17_1.add(-model.ramp_pem * model.P_pem_cap <= model.p_pem[ω,t] - model.p_pem[ω,t-1])

model.c17_2 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c17_2.add(model.p_pem[ω,t] - model.p_pem[ω,t-1] <= model.ramp_pem * model.P_pem_cap)


#model.c25_1 = pe.ConstraintList()
#for t in model.T:
#  model.c25_1.add(model.zT[t] >= -model.p_grid[t]/model.P_grid_cap)

#model.c25_2 = pe.ConstraintList()
#for t in model.T:
#  model.c25_2.add(model.zT[t] <= 1-model.p_grid[t]/model.P_grid_cap)

#model.c25_3 = pe.ConstraintList()
#for t in model.T:
#  model.c25_3.add(model.cT[t] == (1-model.zT[t])*model.CT - model.zT[t]*model.PT)

model.c19_1 = pe.ConstraintList()
for t in model.T:
  model.c19_1.add((R_FCR_min/bidres_FCR)*model.zFCR[t] <= model.bx_FCR[t])

model.c19_2 = pe.ConstraintList()
for t in model.T:
  model.c19_2.add(model.bx_FCR[t] <=(R_FCR_max/bidres_FCR)* model.zFCR[t])

model.c19_3 = pe.ConstraintList()
for t in model.T:
  model.c19_3.add(model.b_FCR[t] == bidres_FCR* model.bx_FCR[t])


model.c19_4 = pe.ConstraintList()
for t in model.T_block:
    model.c19_4.add(model.b_FCR[t+1] == model.b_FCR[t])
    model.c19_4.add(model.b_FCR[t+2] == model.b_FCR[t])
    model.c19_4.add(model.b_FCR[t+3] == model.b_FCR[t]) 


model.c22_1 = pe.ConstraintList()
for t in model.T:
  model.c22_1.add(model.bx_aFRR_up[t] >= (model.R_aFRR_min/model.bidres_aFRR)*model.zaFRRup[t])

model.c22_2 = pe.ConstraintList()
for t in model.T:
  model.c22_2.add(model.bx_aFRR_up[t] <= (model.R_aFRR_max/model.bidres_aFRR)*model.zaFRRup[t])

model.c22_3 = pe.ConstraintList()
for t in model.T:
  model.c22_3.add(model.b_aFRR_up[t] == model.bx_aFRR_up[t]*(model.bidres_aFRR))

model.c23_1 = pe.ConstraintList()
for t in model.T:
  model.c23_1.add(model.bx_aFRR_down[t] >= (model.R_aFRR_min/model.bidres_aFRR)*model.zaFRRdown[t])

model.c23_2 = pe.ConstraintList()
for t in model.T:
  model.c23_2.add(model.bx_aFRR_down[t] <= (model.R_aFRR_max/model.bidres_aFRR)*model.zaFRRdown[t])

model.c23_3 = pe.ConstraintList()
for t in model.T:
  model.c23_3.add(model.b_aFRR_down[t] == model.bx_aFRR_down[t]*(model.bidres_aFRR))

model.c24_1 = pe.ConstraintList()
for t in model.T:
  model.c24_1.add(model.bx_mFRR_up[t] >= (model.R_mFRR_min/model.bidres_mFRR)*model.zmFRRup[t])

model.c24_2 = pe.ConstraintList()
for t in model.T:
  model.c24_2.add(model.bx_mFRR_up[t] <= (model.R_mFRR_max/model.bidres_mFRR)*model.zmFRRup[t])

model.c24_3 = pe.ConstraintList()
for t in model.T:
  model.c24_3.add(model.b_mFRR_up[t] == model.bx_mFRR_up[t]*model.bidres_mFRR)


# grid constraints taking reserves into account
model.c20_1 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c20_1.add(model.P_grid_cap + (model.p_import[ω,t]-model.p_export[ω,t])  >= model.r_FCR[ω,t] + model.r_aFRR_up[ω,t] + model.r_mFRR_up[ω,t])

model.c20_2 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c20_2.add(model.P_grid_cap - (model.p_import[ω,t]-model.p_export[ω,t])  >= model.r_FCR[ω,t] + model.r_aFRR_down[ω,t])

model.c21_1 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c21_1.add(model.P_pem_cap - model.p_pem[ω,t]  >= model.r_FCR[ω,t] + model.r_aFRR_down[ω,t])

model.c21_2 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c21_2.add(model.p_pem[ω,t] - model.P_pem_min >= model.r_FCR[ω,t] + model.r_aFRR_up[ω,t] + model.r_mFRR_up[ω,t])

model.c22 = pe.ConstraintList()
for t in model.T:
  model.c22.add(model.b_FCR[t]*2 + model.b_aFRR_up[t] + model.b_mFRR_up[t] <= model.P_pem_cap - model.P_pem_min)
  model.c22.add(model.b_FCR[t]*2 + model.b_aFRR_down[t] <= model.P_pem_cap - model.P_pem_min)

###############SOLVE THE MODEL########################

#model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
instance = model.create_instance()
results = solver.solve(instance)
print(results)
#instance.display()

#print("Print values for each variable explicitly")
#for i in instance.p_grid:
#  print(str(instance.p_grid[i]), instance.p_grid[i].value)
#  print(str(instance.zT[i]), instance.zT[i].value)
#  print(str(instance.cT[i]), instance.cT[i].value)

#for i in instance.p_PV:
#  print(str(instance.p_PV[i]), instance.p_PV[i].value)
#for i in instance.p_pem:
#  print(str(instance.p_pem[i]), instance.p_pem[i].value)
#for i in instance.r_FCR:
#  print(str(instance.r_FCR[i]), instance.r_FCR[i].value)
#for i in instance.rx_aFRR_up:
#  print(str(instance.rx_aFRR_up[i]), instance.rx_aFRR_up[i].value)
#for i in instance.r_aFRR_up:
#  print(str(instance.r_aFRR_up[i]), instance.r_aFRR_up[i].value)
#for i in instance.zaFRRup:
#  print(str(instance.zaFRRup[i]), instance.zaFRRup[i].value)
#for i in instance.r_aFRR_down:
#  print(str(instance.r_aFRR_down[i]), instance.r_aFRR_down[i].value)
#for i in instance.zaFRRdown:
#  print(str(instance.zaFRRdown[i]), instance.zaFRRdown[i].value)
#for i in instance.r_mFRR_up:
#  print(str(instance.r_mFRR_up[i]), instance.r_mFRR_up[i].value)
#for i in instance.zmFRRup:
#  print(str(instance.zmFRRup[i]), instance.zmFRRup[i].value)


#for i in instance.m_H2:
#  print(str(instance.m_H2[i]), instance.m_H2[i].value)
#for i in instance.m_CO2:
#  print(str(instance.m_CO2[i]), instance.m_CO2[i].value)
#CO2Mass = sum(instance.m_CO2)
#print(CO2Mass)
#for i in instance.m_Ri:
#  print(str(instance.m_Ri[i]), instance.m_Ri[i].value)
#for i in instance.m_Ro:
#  print(str(instance.m_Ro[i]), instance.m_Ro[i].value)
#for i in instance.m_H2O:
#  print(str(instance.m_H2O[i]), instance.m_H2O[i].value)
#for i in instance.m_Pu:
#  print(str(instance.m_Pu[i]), instance.m_Pu[i].value)
#for i in instance.m_Pu:
#  print(str(instance.m_Pu[i]), instance.m_Pu[i].value)

#for i in instance.s_raw:
#  print(str(instance.s_raw[i]), instance.s_raw[i].value)
#for i in instance.s_Pu:
#  print(str(instance.s_Pu[i]), instance.s_Pu[i].value)


#for i in instance.r_FCR:
 # print(str(instance.r_FCR[i]), instance.r_FCR[i].value)


#Converting Pyomo resulst to list
P_PV1 = [instance.p_PV[1,i].value for i in range(1,T+1)]
P_PV2 = [instance.p_PV[2,i].value for i in range(1,T+1)]
P_import1 = [instance.p_import[1,i].value for i in range(1,T+1)]
P_import2 = [instance.p_import[2,i].value for i in range(1,T+1)]
P_export1 = [instance.p_export[1,i].value for i in range(1,T+1)]
P_export2 = [instance.p_export[2,i].value for i in range(1,T+1)]
P_grid1 = [P_import1[i] - P_export1[i] for i in range(0,len(P_import1)) ]
P_grid2 = [P_import2[i] - P_export2[i] for i in range(0,len(P_import2)) ]
m_ri1 = [instance.m_Ri[1,i].value for i in range(1,T+1)]
m_ri2 = [instance.m_Ri[2,i].value for i in range(1,T+1)]
m_ro1 = [instance.m_Pu[1,i].value for i in range(1,T+1)]
m_ro2 = [instance.m_Pu[2,i].value for i in range(1,T+1)]  
m_pu1 = [instance.m_Pu[1,i].value for i in range(1,T+1)]
m_pu2 = [instance.m_Pu[2,i].value for i in range(1,T+1)] 
P_PEM1 = [instance.p_pem[1,i].value for i in range(1,T+1)]  
P_PEM2 = [instance.p_pem[2,i].value for i in range(1,T+1)]  
b_FCR = [instance.b_FCR[i].value for i in range(1,T+1)]
R_FCR1 = [instance.r_FCR[1,i].value for i in range(1,T+1)]
R_FCR2 = [instance.r_FCR[2,i].value for i in range(1,T+1)]
b_mFRRup = [instance.b_mFRR_up[i].value for i in range(1,T+1)]
R_mFRRup1 = [instance.r_mFRR_up[1,i].value for i in range(1,T+1)]
R_mFRRup2= [instance.r_mFRR_up[2,i].value for i in range(1,T+1)]
R_aFRRup1 = [instance.r_aFRR_up[1,i].value for i in range(1,T+1)]
R_aFRRup2 = [instance.r_aFRR_up[2,i].value for i in range(1,T+1)]
R_aFRRdown1 = [instance.r_aFRR_down[1,i].value for i in range(1,T+1)]
R_aFRRdown2 = [instance.r_aFRR_down[2,i].value for i in range(1,T+1)]
z_grid1 = [instance.z_grid[1,i].value for i in range(1,T+1)]
z_grid2 = [instance.z_grid[2,i].value for i in range(1,T+1)]
s_raw1 = [instance.s_raw[1,i].value for i in range(1,T+1)]
s_raw2 = [instance.s_raw[2,i].value for i in range(1,T+1)]
s_pu1 = [instance.s_pu[1,i].value for i in range(1,T+1)]
s_pu2 = [instance.s_pu[2,i].value for i in range(1,T+1)]
#sRaw1 = [instance.s_raw[1,i].value for i in range(1,T+1)]
#sRaw2 = [instance.s_raw[2,i].value for i in range(1,T+1)]  
#sPu1 = [instance.s_Pu[1,i].value for i in range(1,T+1)]  
#sPu2 = [instance.s_Pu[2,i].value for i in range(1,T+1)]  

#Creating result DataFrame
df_results = pd.DataFrame({#Col name : Value(list)
                          'P_PEM1' : P_PEM1,
                          'P_PEM2' : P_PEM2,
                          'P_PV1' : P_PV1,
                          'P_PV2' : P_PV2,
                          'mFRR_up1': R_mFRRup1,
                          'mFRR_up1': R_mFRRup1,
                          'aFRR_up1': R_aFRRup1,
                          'aFRR_up2': R_aFRRup2,
                          'aFRR_down1': R_aFRRdown1,
                          'aFRR_down2': R_aFRRdown2,
                          'Raw Storage1' : s_raw1,
                          'Raw Storage2' : s_raw2,
                          'Pure Storage1' : s_pu1,
                          'Pure Storage2' : s_pu2,
                          'P_grid1' : P_grid1,
                          'P_grid2' : P_grid2,
                          'P_import1' : P_import1,
                          'P_import2' : P_import2,
                          'P_export1' : P_export1,
                          'P_export2' : P_export2,
                          'Raw_In1' : m_ri1,
                          'Raw_In2' : m_ri2,
                          'Raw_Out1' : m_ro1,
                          'Raw_Out2' : m_ro2,
                          'Pure_In1': m_pu1,
                          'Pure_In2': m_pu2,
                          'z_grid1' : z_grid1,
                          'z_grid2' : z_grid2,
                          'DA1' : list(DA.values()),
                          'DA1' : list(DA.values()),
                          'FCR "up"': R_FCR1, 
                          'FCR "up"': R_FCR2, 
                          'FCR "down"': R_FCR1,
                          'FCR "down"': R_FCR2,
                          'cFCR1' : list(c_FCRs.values())[0:168],
                          'cFCR2' : list(c_FCRs.values())[169:337],
                          'aFRRup1' : list(c_aFRR_ups.values())[0:168],
                          'aFRRup2' : list(c_aFRR_ups.values())[169:337],
                          'aFRRdown1' : list(c_aFRR_downs.values())[0:168],
                          'aFRRdown2' : list(c_aFRR_downs.values())[169:337],
                          'mFRRup1' : list(c_mFRR_ups.values())[0:168],
                          'mFRRup2' : list(c_mFRR_ups.values())[169:337],
                          'Demand1' : list(Demand.values()),
                          'Demand2' : list(Demand.values())
                          }, index=DateRange,
                          )




#save to Excel 
df_results.to_excel("Result_files/Model2_All2020.xlsx")




 