#V1 -> V2: Include reserve products in optimization
import pyomo.environ as pe
import pyomo.opt as po
from pyomo.core import *
import pandas as pd 
import numpy as np
from Opt_Constants import *
from Data_process import P_PV_max, DA, Demand, c_FCR, c_aFRR_up, c_aFRR_down, c_mFRR_up, DateRange, pem_setpoint, hydrogen_mass_flow
from Settings import *
import csv
#____________________________________________
solver = po.SolverFactory('gurobi')
model = pe.ConcreteModel()


#set t in T
T = len(P_PV_max)
model.T = pe.RangeSet(1,T)
model.T_block = pe.RangeSet(1,T,4)





#lst = []
#for i in model.T_block:
#  lst.append(i)

#initializing parameters
model.P_PV_max = pe.Param(model.T, initialize=P_PV_max)
model.DA = pe.Param(model.T, initialize=DA)
model.m_demand = pe.Param(model.T, initialize = Demand)
model.c_FCR = pe.Param(model.T, initialize = c_FCR)
model.c_aFRR_up = pe.Param(model.T, initialize = c_aFRR_up)
model.c_aFRR_down = pe.Param(model.T, initialize = c_aFRR_down)
model.c_mFRR_up = pe.Param(model.T, initialize = c_mFRR_up)

model.P_pem_cap = P_pem_cap 
model.P_pem_min = P_pem_min
model.P_com = P_com
model.P_grid_cap = P_grid_cap
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
model.p_import = pe.Var(model.T, domain=pe.NonNegativeReals)
model.p_export = pe.Var(model.T, domain=pe.NonNegativeReals)
model.p_PV = pe.Var(model.T, domain=pe.NonNegativeReals)
model.p_pem = pe.Var(model.T, domain=pe.NonNegativeReals, bounds=(0,52.5))
model.m_H2 = pe.Var(model.T, domain=pe.NonNegativeReals, bounds=(0,1100))
model.m_CO2 = pe.Var(model.T, domain=pe.NonNegativeReals)
model.m_H2O = pe.Var(model.T, domain=pe.NonNegativeReals)
model.m_Ri = pe.Var(model.T, domain=pe.NonNegativeReals)
model.m_Ro = pe.Var(model.T, domain=pe.NonNegativeReals)
model.m_Pu = pe.Var(model.T, domain=pe.NonNegativeReals)
model.s_raw = pe.Var(model.T, domain=pe.NonNegativeReals)
model.s_Pu = pe.Var(model.T, domain=pe.NonNegativeReals)
model.zFCR = pe.Var(model.T, domain = pe.Binary) #Defining the first binary decision variable
model.r_FCR =pe.Var(model.T, domain = pe.NonNegativeReals) #Defining the variable of FCR reserve capacity
model.rx_FCR = pe.Var(model.T, domain = pe.NonNegativeIntegers)
model.rx_aFRR_up = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
model.r_aFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals)
model.zaFRRup = pe.Var(model.T, domain = pe.Binary) #binary decision variable
model.rx_aFRR_down = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
model.r_aFRR_down = pe.Var(model.T, domain = pe.NonNegativeReals)
model.zaFRRdown = pe.Var(model.T, domain = pe.Binary) #binary decision variable
model.rx_mFRR_up = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
model.r_mFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals)
model.zmFRRup = pe.Var(model.T, domain = pe.Binary) #binary decision variable
#model.zT = pe.Var(model.T, domain = pe.Binary) #binary decision variable
#model.cT = pe.Var(model.T, domain = pe.Reals)
model.z_grid = pe.Var(model.T, domain = pe.Binary) #binary decision variable
model.c_obj = pe.Var(model.T, domain = pe.Reals)


#Objective
expr = sum((model.DA[t]+model.CT)*model.p_import[t] - (model.DA[t]-model.PT)*model.p_export[t] - (model.c_FCR[t]*model.r_FCR[t] + model.c_aFRR_up[t]*model.r_aFRR_up[t] + model.c_aFRR_down[t]*model.r_aFRR_down[t] + model.c_mFRR_up[t]*model.r_mFRR_up[t]) for t in model.T)
model.objective = pe.Objective(sense = pe.minimize, expr=expr)

#creating a set of constraints
model.c1 = pe.ConstraintList()
for t in model.T:
    model.c1.add((model.p_import[t]-model.p_export[t]) + model.p_PV[t] == model.p_pem[t] + model.P_com)

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
for t in model.T:
    model.c2.add(model.p_import[t] <= model.z_grid[t]*model.P_grid_cap)
    model.c2.add(model.p_export[t] <= (1-model.z_grid[t])*model.P_grid_cap)


#Constraint 3
model.c3_1 = pe.ConstraintList()
for t in model.T:
    model.c3_1.add(0 <= model.p_PV[t])
#Constraint 3
model.c3_2 = pe.ConstraintList()
for t in model.T:
    model.c3_2.add(model.p_PV[t] <= model.P_PV_max[t])

model.c4_1 = pe.ConstraintList()
for t in model.T:
    model.c4_1.add(model.P_pem_min <= model.p_pem[t])

model.c4_2 = pe.ConstraintList()
for t in model.T:
    model.c4_2.add(model.p_pem[t] <= model.P_pem_cap)

if sEfficiency == 'pw':
  model.c_piecewise = Piecewise(  model.T,
                          model.m_H2,model.p_pem,
                        pw_pts=pem_setpoint,
                        pw_constr_type='EQ',
                        f_rule=hydrogen_mass_flow,
                        pw_repn='SOS2')
                   
if sEfficiency == 'k':
  model.csimple = pe.ConstraintList()
  for t in model.T:
      model.csimple.add(model.p_pem[t] == model.m_H2[t]/model.eff)

model.c6 = pe.ConstraintList()
for t in model.T:
    model.c6.add(model.m_CO2[t] == model.r_in*model.m_H2[t])

model.c7 = pe.ConstraintList()
for t in model.T:
    model.c7.add(model.m_Ri[t] == model.m_H2[t] + model.m_CO2[t])

model.c8 = pe.ConstraintList()
for t in model.T:
    model.c8.add(model.m_Ro[t] == model.m_Pu[t] + model.m_H2O[t])

model.c9 = pe.ConstraintList()
for t in model.T:
    model.c9.add(model.m_Pu[t] == model.r_out * model.m_H2O[t])

model.c10 = pe.ConstraintList()
for t in model.T:
    model.c10.add(model.m_Pu[t] == model.k_d)

#model.c11_1 = pe.ConstraintList()
#for t in model.T:
#    model.c11_1.add(0 <= model.s_raw[t])

model.c11_2 = pe.ConstraintList()
for t in model.T:
    model.c11_2.add(model.s_raw[t] <= model.S_raw_max)

model.c12 = pe.ConstraintList()
for t in model.T:
    if t >= 2:
        model.c12.add(model.s_raw[t] == model.s_raw[t-1] + model.m_Ri[t] - model.m_Ro[t])

model.c13_1 = pe.Constraint(expr=model.s_raw[1] == 0.5*model.S_raw_max + model.m_Ri[1] - model.m_Ro[1])

model.c13_2 = pe.Constraint(expr=0.5*model.S_raw_max == model.s_raw[T])

#model.c14_1 = pe.ConstraintList()
#for t in model.T:
#  model.c14_1.add(0 <= model.s_Pu[t])

model.c14_2 = pe.ConstraintList()
for t in model.T:
  model.c14_2.add(model.s_Pu[t] <= model.S_Pu_max)

model.c15 = pe.Constraint(expr = model.s_Pu[1] == model.m_Pu[1])

model.c16 = pe.ConstraintList()
for t in model.T:
  if t >= 2:
    model.c16.add(model.s_Pu[t] == model.s_Pu[t-1] + model.m_Pu[t] - model.m_demand[t])

model.c17_1 = pe.ConstraintList()
for t in model.T:
  if t >= 2:
    model.c17_1.add(-model.ramp_pem * model.P_pem_cap <= model.p_pem[t] - model.p_pem[t-1])

model.c17_2 = pe.ConstraintList()
for t in model.T:
  if t >= 2:
    model.c17_2.add(model.p_pem[t] - model.p_pem[t-1] <= model.ramp_pem * model.P_pem_cap)


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
  model.c19_1.add(model.r_FCR[t] >= (R_FCR_min/bidres_FCR)*model.zFCR[t])

model.c19_2 = pe.ConstraintList()
for t in model.T:
  model.c19_2.add(model.r_FCR[t] <=(R_FCR_max/bidres_FCR)* model.zFCR[t])

model.c19_3 = pe.ConstraintList()
for t in model.T:
  model.c19_3.add(model.r_FCR[t] == bidres_FCR* model.rx_FCR[t])


model.c19_4 = pe.ConstraintList()
for t in model.T_block:
    model.c19_4.add(model.r_FCR[t+1] == model.r_FCR[t])
    model.c19_4.add(model.r_FCR[t+2] == model.r_FCR[t])
    model.c19_4.add(model.r_FCR[t+3] == model.r_FCR[t]) 


model.c22_1 = pe.ConstraintList()
for t in model.T:
  model.c22_1.add(model.rx_aFRR_up[t] >= (model.R_aFRR_min/model.bidres_aFRR)*model.zaFRRup[t])

model.c22_2 = pe.ConstraintList()
for t in model.T:
  model.c22_2.add(model.rx_aFRR_up[t] <= (model.R_aFRR_max/model.bidres_aFRR)*model.zaFRRup[t])

model.c22_3 = pe.ConstraintList()
for t in model.T:
  model.c22_3.add(model.r_aFRR_up[t] == model.rx_aFRR_up[t]*(model.bidres_aFRR))

model.c22_4 = pe.ConstraintList()
for t in model.T:
  model.c22_4.add(model.r_aFRR_up[t] <= model.R_aFRR_max)

model.c23_1 = pe.ConstraintList()
for t in model.T:
  model.c23_1.add(model.rx_aFRR_down[t] >= (model.R_aFRR_min/model.bidres_aFRR)*model.zaFRRdown[t])

model.c23_2 = pe.ConstraintList()
for t in model.T:
  model.c23_2.add(model.rx_aFRR_down[t] <= (model.R_aFRR_max/model.bidres_aFRR)*model.zaFRRdown[t])

model.c23_3 = pe.ConstraintList()
for t in model.T:
  model.c23_3.add(model.r_aFRR_down[t] == model.rx_aFRR_down[t]*(model.bidres_aFRR))

model.c23_4 = pe.ConstraintList()
for t in model.T:
  model.c23_4.add(model.r_aFRR_down[t] <= model.R_aFRR_max)

model.c24_1 = pe.ConstraintList()
for t in model.T:
  model.c24_1.add(model.rx_mFRR_up[t] >= (model.R_mFRR_min/model.bidres_mFRR)*model.zmFRRup[t])

model.c24_2 = pe.ConstraintList()
for t in model.T:
  model.c24_2.add(model.rx_mFRR_up[t] <= (model.R_mFRR_max/model.bidres_mFRR)*model.zmFRRup[t])

model.c24_3 = pe.ConstraintList()
for t in model.T:
  model.c24_3.add(model.r_mFRR_up[t] == model.rx_mFRR_up[t]*model.bidres_mFRR)

model.c24_4 = pe.ConstraintList()
for t in model.T:
  model.c24_4.add(model.r_mFRR_up[t] <= model.R_mFRR_max)


# grid constraints taking reserves into account
model.c20_1 = pe.ConstraintList()
for t in model.T:
  model.c20_1.add(model.P_grid_cap + (model.p_import[t]-model.p_export[t])  >= model.r_FCR[t] + model.r_aFRR_up[t] + model.r_mFRR_up[t])

model.c20_2 = pe.ConstraintList()
for t in model.T:
  model.c20_2.add(model.P_grid_cap - (model.p_import[t]-model.p_export[t])  >= model.r_FCR[t] + model.r_aFRR_down[t])

model.c21_1 = pe.ConstraintList()
for t in model.T:
  model.c21_1.add(model.P_pem_cap - model.p_pem[t]  >= model.r_FCR[t] + model.r_aFRR_down[t])

model.c21_2 = pe.ConstraintList()
for t in model.T:
  model.c21_2.add(model.p_pem[t] - model.P_pem_min >= model.r_FCR[t] + model.r_aFRR_up[t] + model.r_mFRR_up[t])


model.cObj = pe.ConstraintList()
for t in model.T:
  model.cObj.add(model.c_obj[t] == (model.DA[t]+model.CT)*model.p_import[t] - (model.DA[t]-model.PT)*model.p_export[t] - (model.c_FCR[t]*model.r_FCR[t] + model.c_aFRR_up[t]*model.r_aFRR_up[t] + model.c_aFRR_down[t]*model.r_aFRR_down[t] + model.c_mFRR_up[t]*model.r_mFRR_up[t]))



###############SOLVE THE MODEL########################

#model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
instance = model.create_instance()
results = solver.solve(instance)
print(results)


#Converting Pyomo resulst to list
sRaw = [instance.s_raw[i].value for i in instance.s_raw]  
sPu = [instance.s_Pu[i].value for i in instance.s_Pu]  
P_PV = [instance.p_PV[i].value for i in instance.p_PV] 
P_import = [instance.p_import[i].value for i in instance.p_import]
P_export = [instance.p_export[i].value for i in instance.p_export]
P_grid = [P_import[i] - P_export[i] for i in range(0,len(P_import)) ]
m_ri = [instance.m_Ri[i].value for i in instance.m_Ri]
m_ro = [instance.m_Pu[i].value for i in instance.m_Pu]  
m_pu = [instance.m_Pu[i].value for i in instance.m_Pu]  
m_H2 = [instance.m_H2[i].value for i in instance.m_H2]  
m_CO2 = [instance.m_CO2[i].value for i in instance.m_CO2]  
P_PEM = [instance.p_pem[i].value for i in instance.p_pem]  
R_FCR = [instance.r_FCR[i].value for i in instance.r_FCR]
R_mFRRup = [instance.r_mFRR_up[i].value for i in instance.r_mFRR_up]
R_aFRRup = [instance.r_aFRR_up[i].value for i in instance.r_aFRR_up]
R_aFRRdown = [instance.r_aFRR_down[i].value for i in instance.r_aFRR_down]
z_grid = [instance.z_grid[i].value for i in instance.z_grid]
s_raw = [instance.s_raw[i].value for i in instance.s_raw]
s_pu = [instance.s_Pu[i].value for i in instance.s_Pu]
Obj = [instance.c_obj[i].value for i in instance.c_obj]


#Creating result DataFrame
df_results = pd.DataFrame({#Col name : Value(list)
                          'P_PEM' : P_PEM,
                          'P_grid' : P_grid,
                          'P_import' : P_import,
                          'P_export' : P_export,
                          'z_grid' : z_grid,
                          'P_PV' : P_PV,
                          'c_DA' : list(DA.values()),
                          'r_FCR': R_FCR, 
                          'c_FCR' : list(c_FCR.values()),
                          'r_aFRR_up': R_aFRRup,
                          'c_aFRR_up' : list(c_aFRR_up.values()),
                          'r_aFRR_down': R_aFRRdown,
                          'c_aFRR_down' : list(c_aFRR_down.values()),
                          'r_mFRR_up': R_mFRRup,
                          'c_mFRRup' : list(c_mFRR_up.values()),
                          'Raw_In' : m_ri,
                          'Raw_Out' : m_ro,
                          'Pure_In': m_pu,
                          'm_H2': m_H2,
                          'm_CO2' : m_CO2,
                          'Raw Storage' : sRaw,
                          'Pure Storage' : s_pu,
                          'Demand' : list(Demand.values()),
                          'Objective' : Obj
                          }, index=DateRange,
                          )


#save to Excel 

df_results.to_excel("Result_files/V2_"+Start_date[:10]+"_"+End_date[:10]+ ".xlsx")




## Parameter file ## 

a = [('P_pem_cap', model.P_pem_cap),
    ('P_pem_min', model.P_pem_min),
    ('P_com', model.P_com),
    ('P_grid_cap', model.P_grid_cap),
    ('r_in', model.r_in),
    ('r_out', model.r_out),
    ('k_d', model.k_d),
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
    ('bidres_mFRR', model.bidres_mFRR)]


with open("Result_files/V2_Parameters_"+Start_date[:10]+"_"+End_date[:10]+ ".csv", 'w', newline='') as csvfile:
    my_writer = csv.writer(csvfile,delimiter=',')
    my_writer.writerows(a)

 