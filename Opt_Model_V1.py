#from mmap import MAP_POPULATE
import pyomo.environ as pe
import pyomo.opt as po
from pyomo.core import *
import pandas as pd 
from Opt_Constants import *
from Data_process import P_PV_max, DA, Demand, DateRange, pem_setpoint, hydrogen_mass_flow
from Settings import *
#____________________________________________


solver = po.SolverFactory('gurobi')
model = pe.ConcreteModel()

#set t in T
T = len(P_PV_max)
model.T = pe.RangeSet(1,T)


#initializing parameters
model.P_PV_max = pe.Param(model.T, initialize=P_PV_max)
model.DA = pe.Param(model.T, initialize=DA)
model.m_demand = pe.Param(model.T, initialize = Demand)

model.P_pem_cap = P_pem_cap 
model.P_pem_min = P_pem_min
model.P_com = P_com
model.P_grid_cap = P_grid_cap
model.r_in = r_in
model.r_out = r_out
model.k_d = k_d
model.S_Pu_max = S_Pu_max
model.S_raw_max = S_raw_max
model.m_H2_max = m_H2_max
model.ramp_pem = ramp_pem
model.ramp_com = ramp_com
model.P_PV_cap = P_PV_cap
model.PT = PT
model.CT = CT
#defining variables
model.p_grid = pe.Var(model.T, domain=pe.Reals)
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
model.zT = pe.Var(model.T, domain = pe.Binary) #binary decision variable
model.cT = pe.Var(model.T, domain = pe.Reals)
#model.η = pe.Var(model.T, domain = pe.NonNegativeReals)
#Objective
expr = sum((model.DA[t]+model.cT[t])*model.p_grid[t] for t in model.T)
model.objective = pe.Objective(sense = pe.minimize, expr=expr)

#creating a set of constraints
model.c1 = pe.ConstraintList()
for t in model.T:
    model.c1.add(model.p_grid[t] + model.p_PV[t] == model.p_pem[t] + model.P_com)

#Constraint 2.1
model.c2_1 = pe.ConstraintList()
for t in model.T:
    model.c2_1.add(-model.P_grid_cap <= model.p_grid[t])

#Constraint 2.2
model.c2_2 = pe.ConstraintList()
for t in model.T:
    model.c2_2.add(model.p_grid[t] <= model.P_grid_cap)

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

model.c_piecewise = Piecewise(  model.T,
                        model.m_H2,model.p_pem,
                      pw_pts=pem_setpoint,
                      pw_constr_type='EQ',
                      f_rule=hydrogen_mass_flow,
                      pw_repn='SOS2')


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


model.c25_1 = pe.ConstraintList()
for t in model.T:
  model.c25_1.add(model.zT[t] >= -model.p_grid[t]/model.P_grid_cap)

model.c25_2 = pe.ConstraintList()
for t in model.T:
  model.c25_2.add(model.zT[t] <= 1-model.p_grid[t]/model.P_grid_cap)

model.c25_3 = pe.ConstraintList()
for t in model.T:
  model.c25_3.add(model.cT[t] == (1-model.zT[t])*model.CT - model.zT[t]*model.PT)


#model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
results = solver.solve(model)
print(results)



#Converting Pyomo resulst to list
P_PEM = [model.p_pem[i].value for i in model.p_pem]    
P_sPu = [model.s_Pu[i].value for i in model.s_Pu]  
P_PV = [model.p_PV[i].value for i in model.p_PV]  
P_grid = [model.p_grid[i].value for i in model.p_grid]
m_ri = [model.m_Ri[i].value for i in model.m_Ri] 
m_ro = [model.m_Ro[i].value for i in model.m_Ro]   
m_pu = [model.m_Pu[i].value for i in model.m_Pu]  
m_H2 = [model.m_H2[i].value for i in model.m_H2]
m_CO2 = [model.m_CO2[i].value for i in model.m_CO2]
m_H2O = [model.m_H2O[i].value for i in model.m_H2O]
m_H2 = [model.m_H2[i].value for i in model.m_H2]
zT = [model.zT[i].value for i in model.zT]
s_raw = [model.s_raw[i].value for i in model.s_raw]
s_pu = [model.s_Pu[i].value for i in model.s_Pu]





#Creating result DataFrame
df_results = pd.DataFrame({#Col name : Value(list)
                          'P_PEM' : P_PEM,
                          'P_PV' : P_PV,
                          'P_grid' : P_grid,
                          'Raw_In' : m_ri,
                          'Raw Out': m_ro,
                          'Pure_In': m_pu,
                          'Raw Storage' : s_raw,
                          'Pure Storage' : s_pu,
                          'DA' : list(DA.values()),
                          'm_H2': m_H2,
                          'm_CO2' : m_CO2,
                          'm_H2O' : m_H2O,
                          'Demand' : list(Demand.values()), 
                          'zT' : zT
                          }, index=DateRange,

                          )




#save to Excel 
df_results.to_excel("Result_files/Model_2021.xlsx")


