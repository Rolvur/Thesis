import pyomo.environ as pe
import pyomo.opt as po

from Opt_Constants import *
#____________________________________________



P_PV_max = {1:      0, 
            2:      0, 
            3:      0, 
            4:      0, 
            5:      0, 
            6:     10, 
            7:     50, 
            8:    120, 
            9:    160, 
            10:   120, 
            11:    80, 
            12:   150, 
            13:   180, 
            14:   170, 
            15:   140, 
            16:   105, 
            17:    75, 
            18:    60, 
            19:    45, 
            20:    35, 
            21:     5, 
            22:     0, 
            23:     0, 
            24:     0 }

DA  ={  1:      141.09, 
        2:      135.29, 
        3:      139.29, 
        4:      139.92, 
        5:      139.50, 
        6:      143.12, 
        7:      150.10, 
        8:      180.74, 
        9:      232.10, 
        10:     203.35, 
        11:     193.51, 
        12:     179.28, 
        13:     125.88, 
        14:     138.55, 
        15:     122.01, 
        16:     135.23, 
        17:     152.35, 
        18:     198.51, 
        19:     230.10, 
        20:     247.85, 
        21:     198.58, 
        22:     165.68, 
        23:     162.95, 
        24:     153.60
    }

m_demand  ={1:      0, 
            2:      0, 
            3:      0, 
            4:      0, 
            5:      0, 
            6:      0, 
            7:      0, 
            8:      0, 
            9:      0, 
            10:     0, 
            11:     0, 
            12:     0, 
            13:     0, 
            14:     0, 
            15:     0, 
            16:     0, 
            17:     0, 
            18:     0, 
            19:     0, 
            20:     0, 
            21:     0, 
            22:     0, 
            23:     0, 
            24:     600000
    }


solver = po.SolverFactory('gurobi')



model = pe.ConcreteModel()

#set t in T
model.T = pe.RangeSet(1,24)


#initializing parameters
model.P_PV_max = pe.Param(model.T, initialize=P_PV_max)
model.DA = pe.Param(model.T, initialize=DA)
model.m_demand = pe.Param(model.T, initialize = m_demand)

model.P_pem_cap = P_pem_cap 
model.P_pem_min = P_pem_min
model.P_com = P_com
model.P_H2O = P_H2O
model.P_grid_cap = P_grid_cap
model.k_CR = k_CR
model.r_in = r_in
model.r_out = r_out
model.k_d = k_d
model.S_pure_max = S_pure_max
model.S_raw_max = S_raw_max
model.ramp_pem = ramp_pem
model.ramp_com = ramp_com
model.P_PV_cap = P_PV_cap

#defining variables
model.p_grid = pe.Var(model.T, domain=pe.Reals)
model.p_PV = pe.Var(model.T, domain=pe.Reals)
model.p_pem = pe.Var(model.T, domain=pe.Reals)
model.m_H2 = pe.Var(model.T, domain=pe.Reals)
model.m_CO2 = pe.Var(model.T, domain=pe.Reals)
model.m_Ri = pe.Var(model.T, domain=pe.Reals)
model.S_raw = pe.Var(model.T, domain=pe.Reals)
model.S_pure = pe.Var(model.T, domain=pe.Reals)

expr = sum(model.DA[t]*model.p_grid[t] for t in model.T)
model.objective = pe.Objective(sense = pe.minimize, expr=expr)

#creating a set of constraints
model.c1 = pe.ConstraintList()
for t in model.T:
    lhs = model.p_grid[t] + model.p_PV[t]
    rhs = model.p_pem[t] + model.P_com + model.P_H2O
    model.c1.add(lhs == rhs)

#Constraint 2.1
model.c2_1 = pe.ConstraintList()
for t in model.T:
    lhs = -model.P_grid_cap
    rhs = model.p_grid[t]
    model.c2_1.add(lhs <= rhs)
#Constraint 2.2
model.c2_2 = pe.ConstraintList()
for t in model.T:
    rhs = model.P_grid_cap
    lhs = model.p_grid[t]
    model.c2_2.add(lhs <= rhs)

#Constraint 3
model.c3_1 = pe.ConstraintList()
for t in model.T:
    lhs = 0
    rhs = model.p_PV[t]
    model.c3_1.add(lhs <= rhs)
#Constraint 3
model.c3_2 = pe.ConstraintList()
for t in model.T:
    rhs = model.P_PV_max[t]
    lhs = model.p_PV[t]
    model.c3_2.add(lhs <= rhs)


solver = po.SolverFactory('gurobi')
results = solver.solve(model)
print(results)

print(model.p_grid.values)
print(model.P_grid_cap)

print("Print values for each variable explicitly")
for i in model.p_grid:
  print(str(model.p_grid[i]), model.p_grid[i].value)
for i in model.p_PV:
  print(str(model.p_PV[i]), model.p_PV[i].value)
for i in model.p_pem:
  print(str(model.p_pem[i]), model.p_pem[i].value)
for i in model.p_pem:
  print(str('P_com'), model.P_com)
for i in model.p_pem:
  print(str('P_H2O'), model.P_H2O)

