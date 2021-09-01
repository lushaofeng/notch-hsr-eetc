from gurobipy import *
from xlrd import open_workbook
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# Parameters #
D = 40000  # m
T_total = 670  # s
V_lim_i = 280  # kph
N = 20
J = 31
P_con = 0.99
M = 890000  # kg
K_1 = 10
K_2 = 7
Acc_max_a = 1.2  # m/s2
Acc_max_d = 1.2  # m/s2
n_t = 0.9
n_b = 0.6
A = 5200  # N
B = 38  # N/(km/h)
C = 1.12  # N/(km2/h2)
# Equation (1): Discretization
delta_d = D / N
g = 9.8  # m/s2
Np1 = N + 1
i = list(range(1, Np1))  # 1, 2, ..., 20
ii = list(range(0, Np1))  # 0, 1, ..., 20

# Obtain original notch data from excel #
T_ = {}  # store traction characteristics
B_ = {}  # store braking characteristics
x = []
y = []
wb = open_workbook('Notch characteristics.xls')
sheet = wb.sheets()
# traction characteristic
s = sheet[0]
notch = 1
for col in range(s.ncols):
    if col % 2 == 0:
        for row in range(s.nrows):
            x.append(s.cell(row, col).value)
    else:
        for row in range(s.nrows):
            y.append(s.cell(row, col).value)
        # delete word data
        for index in range(2):
            del x[0]
            del y[0]
        # delete none from the data
        if s.cell(row, col).value == '':  # final element is none
            cutx = x.index('')  # find the start of none
            t_x = x[0: cutx]  # delete none
            cuty = y.index('')
            t_y = y[0: cuty]
        else:
            t_x = x  # longest data
            t_y = y
        cur_t = {('T' + str(notch)): [t_x, t_y]}
        T_.update(cur_t)
        x = []
        y = []
        notch = notch + 1
# braking characteristic
s = sheet[1]
notch = 1
for col in range(s.ncols):
    if col % 2 == 0:
        for row in range(s.nrows):
            x.append(s.cell(row, col).value)
    else:
        for row in range(s.nrows):
            y.append(s.cell(row, col).value)
        # delete word data
        for index in range(2):
            del x[0]
            del y[0]
        # delete none from the data
        if s.cell(row, col).value == '':
            cutx = x.index('')
            b_x = x[0: cutx]
            cuty = y.index('')
            b_y = y[0: cuty]
        else:
            b_x = x
            b_y = y
        cur_b = {('B'+ str(notch)): [b_x, b_y]}
        B_.update(cur_b)
        x = []
        y = []
        notch = notch + 1

# Calculate PWL nodes according to the original data #
N_t = {}  # store the PWL nodes for traction
N_b = {}  # store the PWL nodes for braking
V_min = 1
V_max = 96
# Equation (16): Piecewise linearisation accuracy
delta = (V_max - V_min) / (J - 1)
PWL_SPE = [V_min]
pre = 0
for index in range(J-1):
    pre = pre + delta
    PWL_SPE.append(pre)
# 0~350 km/h
# insert points into T10
inter_T10 = interpolate.interp1d(T_['T10'][0][0:2], T_['T10'][1][0:2], kind='slinear')
inter_T10x = list(np.linspace(T_['T10'][0][0], T_['T10'][0][1], 50))
inter_T10y = list(inter_T10(inter_T10x))
newT10x = inter_T10x + T_['T10'][0][2:]
newT10y = inter_T10y + T_['T10'][1][2:]
for index in T_:
    if index in ['T1', 'T10']:
        inter = interpolate.interp1d(T_[index][0], T_[index][1], kind='slinear')
        inter_y = inter(np.array(PWL_SPE)*3.6)
        cur_notch = {'PWL_' + str(index): inter_y}
        N_t.update(cur_notch)
    else:
        # find the cut part of T10 to connect with the other traction notch
        curve_start = T_[index][0][0]
        for notch in newT10x:
            if notch >= curve_start:
                cut_T10_index = newT10x.index(notch)
                break
        # cut T10
        T10_cutx = newT10x[0: cut_T10_index]
        T10_cuty = newT10y[0: cut_T10_index]
        # connect T10 and Tx
        newx = T10_cutx + T_[index][0]
        newy = T10_cuty + T_[index][1]
        inter = interpolate.interp1d(newx, newy, kind='slinear')
        inter_y = inter(np.array(PWL_SPE)*3.6)
        cur_notch = {'PWL_' + str(index): inter_y}
        N_t.update(cur_notch)
# 0~350 km/h
for index in B_:
    inter = interpolate.interp1d(B_[index][0], B_[index][1], kind='slinear')
    inter_y = inter(np.array(PWL_SPE)*3.6)
    cur_notch = {'PWL_' + str(index): inter_y}
    N_b.update(cur_notch)

# Altitude information #
delta_H_i = []
for index in i:
    delta_H_i.append(0)

# Modelling #
m = Model('Case study 1: feasibility validation of the model on an artificial line')
# set variables in discrete segments
delta_t_i = m.addVars(i, vtype=GRB.CONTINUOUS, name="Elapsed time")
f_i = m.addVars(i, vtype=GRB.CONTINUOUS, name="Average drag force")
E_i = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Energy consumption")
v_ave_i = m.addVars(i, vtype=GRB.CONTINUOUS, name="Average speed")
v_ave_i2 = m.addVars(i, vtype=GRB.CONTINUOUS, name="Square of average speed")
v_ave_i1d = m.addVars(i, vtype=GRB.CONTINUOUS, name="1/v_ave_i")
F_i = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Applied force')
F_min_i = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Maximum applied force')
F_max_i = m.addVars(i, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Minimum applied force')
k = m.addVars(i, lb=-7, ub=10, vtype=GRB.INTEGER, name='Notch index')
# Approximation characteristic functions of k^th notch
f_1 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Traction characteristic of Notch T1')
f_2 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Traction characteristic of Notch T2')
f_3 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Traction characteristic of Notch T3')
f_4 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Traction characteristic of Notch T4')
f_5 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Traction characteristic of Notch T5')
f_6 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Traction characteristic of Notch T6')
f_7 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Traction characteristic of Notch T7')
f_8 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Traction characteristic of Notch T8')
f_9 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Traction characteristic of Notch T9')
f_10 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Traction characteristic of Notch T10')
f_b1 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Braking characteristic of Notch B1')
f_b2 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Braking characteristic of Notch B2')
f_b3 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Braking characteristic of Notch B3')
f_b4 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Braking characteristic of Notch B4')
f_b5 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Braking characteristic of Notch B5')
f_b6 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Braking characteristic of Notch B6')
f_b7 = m.addVars(i, vtype=GRB.CONTINUOUS, name='Braking characteristic of Notch B7')
# binary variables
lambda_1_i = m.addVars(i, vtype=GRB.BINARY, name='Notch T1')
lambda_2_i = m.addVars(i, vtype=GRB.BINARY, name='Notch T2')
lambda_3_i = m.addVars(i, vtype=GRB.BINARY, name='Notch T3')
lambda_4_i = m.addVars(i, vtype=GRB.BINARY, name='Notch T4')
lambda_5_i = m.addVars(i, vtype=GRB.BINARY, name='Notch T5')
lambda_6_i = m.addVars(i, vtype=GRB.BINARY, name='Notch T6')
lambda_7_i = m.addVars(i, vtype=GRB.BINARY, name='Notch T7')
lambda_8_i = m.addVars(i, vtype=GRB.BINARY, name='Notch T8')
lambda_9_i = m.addVars(i, vtype=GRB.BINARY, name='Notch T9')
lambda_10_i = m.addVars(i, vtype=GRB.BINARY, name='Notch T10')
lambda_0_i = m.addVars(i, vtype=GRB.BINARY, name='Coasting')
lambda_b1_i = m.addVars(i, vtype=GRB.BINARY, name='Notch B1')
lambda_b2_i = m.addVars(i, vtype=GRB.BINARY, name='Notch B2')
lambda_b3_i = m.addVars(i, vtype=GRB.BINARY, name='Notch B3')
lambda_b4_i = m.addVars(i, vtype=GRB.BINARY, name='Notch B4')
lambda_b5_i = m.addVars(i, vtype=GRB.BINARY, name='Notch B5')
lambda_b6_i = m.addVars(i, vtype=GRB.BINARY, name='Notch B6')
lambda_b7_i = m.addVars(i, vtype=GRB.BINARY, name='Notch B7')
# set variables at discrete points
v_i = m.addVars(ii, lb=0.0, vtype=GRB.CONTINUOUS, name="Candidate speed")
v_i2 = m.addVars(ii, lb=0.0, vtype=GRB.CONTINUOUS, name="Square of candidate speed")

# set constraints
# Equation (18) & (23): SOS2 variables property 1
alpha = m.addVars(Np1, J, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='Speed-related SOS2 variables')
beta = m.addVars(N, J, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='Average-speed-related SOS2 variables')
for index in ii:
    m.addSOS(GRB.SOS_TYPE2, [alpha[index, j] for j in range(J)])
for index in i:
    m.addSOS(GRB.SOS_TYPE2, [beta[index-1, j] for j in range(J)])

# Equation (2): speed limit
for index in range(1, N):
    m.addConstr(v_i[index] <= V_lim_i / 3.6)
m.addConstr(v_i[0] == 1, name="Departure")
m.addConstr(v_i[N] == 1, name="Pull in")

# Equation (3): Davis equation
for index in i:
    m.addConstr(f_i[index] == A + B * v_ave_i[index] * 3.6 + C * v_ave_i2[index] * 3.6**2, name='Davis equation')

# Equation (4): Average speed
m.addConstrs((v_i[index] + v_i[index-1] - 2 * v_ave_i[index] == 0 for index in i), name='For v_ave_i')

# Equation (5): Travel time
for index in i:
    m.addConstr(delta_t_i[index] == delta_d * v_ave_i1d[index], name='For Δt_i')
m.addConstr(quicksum(delta_t_i) <= T_total, name='Total travel time')

# Equation (6): Principle of conservation of energy
for index in i:
    m.addConstr(F_i[index] * delta_d - 0.5 * M * (v_i2[index] - v_i2[index - 1]) - delta_d * f_i[index] - M * g * delta_H_i[index - 1] == 0, name='Principle of conservation of energy')

# Equation (7): riding comfort
for index in i:
    m.addRange(0.5 * (v_i2[index] - v_i2[index-1]) / delta_d, -Acc_max_d, Acc_max_a, name='Maximum acceleration/deceleration')

# Equation (8) & (9): traction efficiency & regenerative braking effciency
for index in i:
    m.addConstr(E_i[index] * n_t >= F_i[index] * delta_d, name='Traction energy consumption')
    m.addConstr(E_i[index] / n_b >= F_i[index] * delta_d, name='Braking energy recovery')

# Equation (10): Net energy consumption
E_total = quicksum(E_i)

# Equation (11): Notch selection
m.addConstrs((lambda_1_i[index] + lambda_2_i[index] + lambda_3_i[index] + lambda_4_i[index] + lambda_5_i[index]
             + lambda_6_i[index] + lambda_7_i[index] + lambda_8_i[index] + lambda_9_i[index] + lambda_10_i[index]
             + lambda_0_i[index] + lambda_b1_i[index] + lambda_b2_i[index] + lambda_b3_i[index] + lambda_b4_i[index]
             + lambda_b5_i[index] + lambda_b6_i[index] + lambda_b7_i[index] == 1 for index in i), name='Notch selection')

# Equation (13): Maximum applied force
for index in i:
    m.addConstr((lambda_0_i[index] == 1) >> (F_i[index] == 0), name='Coasting operation')
    m.addConstr((lambda_1_i[index] == 1) >> (F_max_i[index] == f_1[index]), name='Maximum applied force of Notch T1')
    m.addConstr((lambda_2_i[index] == 1) >> (F_max_i[index] == f_2[index]), name='Maximum applied force of Notch T2')
    m.addConstr((lambda_3_i[index] == 1) >> (F_max_i[index] == f_3[index]), name='Maximum applied force of Notch T3')
    m.addConstr((lambda_4_i[index] == 1) >> (F_max_i[index] == f_4[index]), name='Maximum applied force of Notch T4')
    m.addConstr((lambda_5_i[index] == 1) >> (F_max_i[index] == f_5[index]), name='Maximum applied force of Notch T5')
    m.addConstr((lambda_6_i[index] == 1) >> (F_max_i[index] == f_6[index]), name='Maximum applied force of Notch T6')
    m.addConstr((lambda_7_i[index] == 1) >> (F_max_i[index] == f_7[index]), name='Maximum applied force of Notch T7')
    m.addConstr((lambda_8_i[index] == 1) >> (F_max_i[index] == f_8[index]), name='Maximum applied force of Notch T8')
    m.addConstr((lambda_9_i[index] == 1) >> (F_max_i[index] == f_9[index]), name='Maximum applied force of Notch T9')
    m.addConstr((lambda_10_i[index] == 1) >> (F_max_i[index] == f_10[index]), name='Maximum applied force of Notch T10')
    m.addConstr((lambda_b1_i[index] == 1) >> (F_min_i[index] == -f_b1[index]), name='Minimum applied force of Notch B1')
    m.addConstr((lambda_b2_i[index] == 1) >> (F_min_i[index] == -f_b2[index]), name='Minimum applied force of Notch B2')
    m.addConstr((lambda_b3_i[index] == 1) >> (F_min_i[index] == -f_b3[index]), name='Minimum applied force of Notch B3')
    m.addConstr((lambda_b4_i[index] == 1) >> (F_min_i[index] == -f_b4[index]), name='Minimum applied force of Notch B4')
    m.addConstr((lambda_b5_i[index] == 1) >> (F_min_i[index] == -f_b5[index]), name='Minimum applied force of Notch B5')
    m.addConstr((lambda_b6_i[index] == 1) >> (F_min_i[index] == -f_b6[index]), name='Minimum applied force of Notch B6')
    m.addConstr((lambda_b7_i[index] == 1) >> (F_min_i[index] == -f_b7[index]), name='Minimum applied force of Notch B7')

# Equation (14): Minimum applied force
for index in i:
    m.addConstr((lambda_1_i[index] == 1) >> (F_min_i[index] == f_1[index] * (1 - P_con)), name='Minimum applied force of Notch T1')
    m.addConstr((lambda_2_i[index] == 1) >> (F_min_i[index] == f_1[index] + (f_2[index] - f_1[index]) * (1 - P_con)), name='Minimum applied force of Notch T2')
    m.addConstr((lambda_3_i[index] == 1) >> (F_min_i[index] == f_2[index] + (f_3[index] - f_2[index]) * (1 - P_con)), name='Minimum applied force of Notch T3')
    m.addConstr((lambda_4_i[index] == 1) >> (F_min_i[index] == f_3[index] + (f_4[index] - f_3[index]) * (1 - P_con)), name='Minimum applied force of Notch T4')
    m.addConstr((lambda_5_i[index] == 1) >> (F_min_i[index] == f_4[index] + (f_5[index] - f_4[index]) * (1 - P_con)), name='Minimum applied force of Notch T5')
    m.addConstr((lambda_6_i[index] == 1) >> (F_min_i[index] == f_5[index] + (f_6[index] - f_5[index]) * (1 - P_con)), name='Minimum applied force of Notch T6')
    m.addConstr((lambda_7_i[index] == 1) >> (F_min_i[index] == f_6[index] + (f_7[index] - f_6[index]) * (1 - P_con)), name='Minimum applied force of Notch T7')
    m.addConstr((lambda_8_i[index] == 1) >> (F_min_i[index] == f_7[index] + (f_8[index] - f_7[index]) * (1 - P_con)), name='Minimum applied force of Notch T8')
    m.addConstr((lambda_9_i[index] == 1) >> (F_min_i[index] == f_8[index] + (f_9[index] - f_8[index]) * (1 - P_con)), name='Minimum applied force of Notch T9')
    m.addConstr((lambda_10_i[index] == 1) >> (F_min_i[index] == f_9[index] + (f_10[index] - f_9[index]) * (1 - P_con)), name='Minimum applied force of Notch T10')
    m.addConstr((lambda_b1_i[index] == 1) >> (F_max_i[index] == -f_b1[index] * (1 - P_con)), name='Maximum applied force of Notch B1')
    m.addConstr((lambda_b2_i[index] == 1) >> (F_max_i[index] == -f_b1[index] - (f_b2[index] - f_b1[index]) * (1 - P_con)), name='Maximum applied force of Notch B2')
    m.addConstr((lambda_b3_i[index] == 1) >> (F_max_i[index] == -f_b2[index] - (f_b3[index] - f_b2[index]) * (1 - P_con)), name='Maximum applied force of Notch B3')
    m.addConstr((lambda_b4_i[index] == 1) >> (F_max_i[index] == -f_b3[index] - (f_b4[index] - f_b3[index]) * (1 - P_con)), name='Maximum applied force of Notch B4')
    m.addConstr((lambda_b5_i[index] == 1) >> (F_max_i[index] == -f_b4[index] - (f_b5[index] - f_b4[index]) * (1 - P_con)), name='Maximum applied force of Notch B5')
    m.addConstr((lambda_b6_i[index] == 1) >> (F_max_i[index] == -f_b5[index] - (f_b6[index] - f_b5[index]) * (1 - P_con)), name='Maximum applied force of Notch B6')
    m.addConstr((lambda_b7_i[index] == 1) >> (F_max_i[index] == -f_b6[index] - (f_b7[index] - f_b6[index]) * (1 - P_con)), name='Maximum applied force of Notch B7')

# Equation (15): Vertical relaxation
for index in i:
    m.addConstr(F_i[index] <= F_max_i[index], name='Maximum applied force')
    m.addConstr(F_i[index] >= F_min_i[index], name='Minimum applied force')

# Piecewise linearisation
# Equation (17): Average speed
for index in i:
    m.addConstr(v_ave_i[index] == (quicksum(PWL_SPE[j] * beta[index-1, j] for j in range(J))), name='Average speed')

# Equation (19) & (24): SOS2 variables property 2
m.addConstrs((alpha.sum(index, '*') == 1 for index in ii), name="SOS2 property_α")
m.addConstrs((beta.sum(index-1, '*') == 1 for index in i), name="SOS2 property_β")

# Equation (20): PWL of applied force
for index in i:
    m.addConstr(f_1[index] == (quicksum(N_t['PWL_T1'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch T1')
    m.addConstr(f_2[index] == (quicksum(N_t['PWL_T2'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch T2')
    m.addConstr(f_3[index] == (quicksum(N_t['PWL_T3'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch T3')
    m.addConstr(f_4[index] == (quicksum(N_t['PWL_T4'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch T4')
    m.addConstr(f_5[index] == (quicksum(N_t['PWL_T5'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch T5')
    m.addConstr(f_6[index] == (quicksum(N_t['PWL_T6'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch T6')
    m.addConstr(f_7[index] == (quicksum(N_t['PWL_T7'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch T7')
    m.addConstr(f_8[index] == (quicksum(N_t['PWL_T8'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch T8')
    m.addConstr(f_9[index] == (quicksum(N_t['PWL_T9'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch T9')
    m.addConstr(f_10[index] == (quicksum(N_t['PWL_T10'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch T10')
    m.addConstr(f_b1[index] == (quicksum(N_b['PWL_B1'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch B1')
    m.addConstr(f_b2[index] == (quicksum(N_b['PWL_B2'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch B2')
    m.addConstr(f_b3[index] == (quicksum(N_b['PWL_B3'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch B3')
    m.addConstr(f_b4[index] == (quicksum(N_b['PWL_B4'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch B4')
    m.addConstr(f_b5[index] == (quicksum(N_b['PWL_B5'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch B5')
    m.addConstr(f_b6[index] == (quicksum(N_b['PWL_B6'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch B6')
    m.addConstr(f_b7[index] == (quicksum(N_b['PWL_B7'][j] * beta[index-1, j] * 1000 for j in range(J))), name='PWL of Notch B7')

# Equation (21): PWL of 1/v_ave_i
for index in i:
    m.addConstr(v_ave_i1d[index] == (quicksum(1 / PWL_SPE[j] * beta[index-1, j] for j in range(J))), name='PWL of 1/v_ave_i')

# Equation (22): PWL of v_ave_i2
for index in i:
    m.addConstr(v_ave_i2[index] == (quicksum(PWL_SPE[j]**2 * beta[index-1, j] for j in range(J))), name='PWL of v_ave_i2')

# Equation (25): Candidate speed
for index in ii:
    m.addConstr(v_i[index] == (quicksum(PWL_SPE[j] * alpha[index, j] for j in range(J))), name='Candidate speed')

# Equation (26): PWL of v_i2
for index in ii:
    m.addConstr(v_i2[index] == (quicksum(PWL_SPE[j]**2 * alpha[index, j] for j in range(J))), name='PWL of v_i2')

# Notch selection information
for index in i:
    m.addConstr((lambda_1_i[index] == 1) >> (k[index] == 1), name='T1')
    m.addConstr((lambda_2_i[index] == 1) >> (k[index] == 2), name='T2')
    m.addConstr((lambda_3_i[index] == 1) >> (k[index] == 3), name='T3')
    m.addConstr((lambda_4_i[index] == 1) >> (k[index] == 4), name='T4')
    m.addConstr((lambda_5_i[index] == 1) >> (k[index] == 5), name='T5')
    m.addConstr((lambda_6_i[index] == 1) >> (k[index] == 6), name='T6')
    m.addConstr((lambda_7_i[index] == 1) >> (k[index] == 7), name='T7')
    m.addConstr((lambda_8_i[index] == 1) >> (k[index] == 8), name='T8')
    m.addConstr((lambda_9_i[index] == 1) >> (k[index] == 9), name='T9')
    m.addConstr((lambda_10_i[index] == 1) >> (k[index] == 10), name='T10')
    m.addConstr((lambda_0_i[index] == 1) >> (k[index] == 0), name='T0')
    m.addConstr((lambda_b1_i[index] == 1) >> (k[index] == -1), name='B1')
    m.addConstr((lambda_b2_i[index] == 1) >> (k[index] == -2), name='B2')
    m.addConstr((lambda_b3_i[index] == 1) >> (k[index] == -3), name='B3')
    m.addConstr((lambda_b4_i[index] == 1) >> (k[index] == -4), name='B4')
    m.addConstr((lambda_b5_i[index] == 1) >> (k[index] == -5), name='B5')
    m.addConstr((lambda_b6_i[index] == 1) >> (k[index] == -6), name='B6')
    m.addConstr((lambda_b7_i[index] == 1) >> (k[index] == -7), name='B7')

# Objective function
m.setObjective(E_total, GRB.MINIMIZE)
m.Params.MIPGap = 0.00
m.write("MILP.lp")
m.optimize()
QP = m.getAttr(GRB.Attr.IsQP)
QCP = m.getAttr(GRB.Attr.IsQCP)
if QP == 1:
    QP = 'Yes.'
else:
    QP = 'No.'
if QCP == 1:
    QCP = 'Yes.'
else:
    QCP = 'No.'
print('Is it QP?', QP)
print('Is it QCP?', QCP)

# Data
# notch curve

Distance_segment_plot = [0]
for index in i:
    Distance_segment_plot.append(index * delta_d / 1000)
    Distance_segment_plot.append(index * delta_d / 1000)
Distance_segment_plot.pop()
Notches = []
for index in i:
    Notches.append(k[index].x)
    Notches.append(k[index].x)
# speed trajectory
v_i_plot = []
for index in ii:
    v_i_plot.append(v_i[index].x * 3.6)
Distance_boundary_plot = []
for index in ii:
    Distance_boundary_plot.append(delta_d * index / 1000)
# Drag force
Drag_force = []
for index in i:
    Drag_force.append(f_i[index].x/1000)
    Drag_force.append(f_i[index].x/1000)
# Maximum force of T10/B7
F_max_T10 = []
F_max_B7 = []
for index in i:
    F_max_B7.append(-f_b7[index].x/1000)
    F_max_B7.append(-f_b7[index].x/1000)
    F_max_T10.append(f_10[index].x/1000)
    F_max_T10.append(f_10[index].x/1000)
# Applied force
F_i_plot = []
for index in i:
    F_i_plot.append(F_i[index].x/1000)
    F_i_plot.append(F_i[index].x/1000)
PWL_SPE = list(np.array(PWL_SPE)*3.6)

# Plot
# Figure 1
Fig1 = plt.figure(1)
plt.grid(linestyle='-.', linewidth='0.5')
plt.xticks(np.linspace(0, 40, 11))
plt.yticks(np.linspace(-8, 10, 10))
plt.xlabel("Distance (km)", fontsize=12)
plt.ylabel('Notch', fontsize=12)
plt.xlim(0, 40)
plt.ylim(-7.5, 10.5)
Notch_curve = plt.plot(Distance_segment_plot, Notches, color='midnightblue', linestyle='--', linewidth='1.5', label='Selected notches')
ax1 = plt.twinx()
ax1.set_ylabel('Speed (km/h)', fontsize=12)
ax1.set_ylim(0, 300)
Speed_trajectory = ax1.plot(Distance_boundary_plot, v_i_plot, color='darkgreen', linewidth='1.5', label='Speed trajectory')
Curves = Notch_curve + Speed_trajectory
labs = [l.get_label() for l in Curves]
ax1.legend(Curves, labs, loc=0, fontsize=12)
Fig1.show()
#Fig1.savefig("Figure1.pdf")#

# Figure 2
Fig2 = plt.figure(2)
plt.grid(linestyle='-.', linewidth='0.5')
plt.xticks(np.linspace(0, 40, 11))
plt.yticks(np.linspace(-600, 600, 5))
plt.xlabel("Distance (km)", fontsize=12)
plt.ylabel('Force (kN)', fontsize=12)
plt.xlim(0, 40)
plt.ylim(-650, 600)
plt.plot(Distance_segment_plot, Drag_force, 'k--', linewidth='1.5', label='Resistance')
plt.plot(Distance_segment_plot, F_max_T10, color='maroon', linewidth='1.5', linestyle='-.', label='Applied force from T10/B7')
plt.plot(Distance_segment_plot, F_max_B7, color='maroon', linewidth='1.5', linestyle='-.')
plt.plot(Distance_segment_plot, F_i_plot, color='k', linewidth='1.5', label='Applied force')
plt.legend(loc=0, fontsize=12)
Fig2.show()
#Fig2.savefig("Figure2.pdf")

# Figure 3
Fig3 = plt.figure(3)
for index in range(1, 5):
    Notch_current = round(k[index].x)
    plt.subplot(2, 2, index)
    plt.xticks(np.linspace(0, 300, 4))
    plt.yticks(np.linspace(-600, 600, 5))
    plt.xlim(0, 350)
    plt.xlabel("Speed (km/h)", fontsize=12)
    plt.ylabel('Force (kN)', fontsize=12)
    plt.grid(linestyle='-.', linewidth='0.5')
    if Notch_current == 1:
        plt.ylim(0, 550)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='1.5')
        for j in N_t:
            plt.plot(PWL_SPE, N_t[index], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, N_t['PWL_T1'], color='darkgreen', linewidth='1.5')
    elif Notch_current > 0:
        plt.ylim(0, 550)
        plt.plot(PWL_SPE, N_t['PWL_T' + str(Notch_current - 1)], color='darkgreen', linewidth='1.5')
        for j in N_t:
            plt.plot(PWL_SPE, N_t[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, N_t['PWL_T' + str(Notch_current)], color='darkgreen', linewidth='1.5')
    elif Notch_current == 0:
        plt.ylim(-300, 300)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='1.5')
    elif Notch_current == -1:
        plt.ylim(-800, 0)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='0.75')
        for j in N_b:
            plt.plot(PWL_SPE, -N_b[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, -N_b['PWL_B1'], color='darkgreen', linewidth='1.5')
    else:
        plt.ylim(-800, 0)
        plt.plot(PWL_SPE, -N_b['PWL_B' + str(-Notch_current - 1)], color='darkgreen', linewidth='1.5')
        for j in N_b:
            plt.plot(PWL_SPE, -N_b[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, -N_b['PWL_B' + str(-Notch_current)], color='darkgreen', linewidth='1.5')
    plt.plot(v_ave_i[index].x * 3.6, F_i[index].x / 1000, 'go')

# Figure 4
Fig4 = plt.figure(4)
for index in range(5, 9):
    Notch_current = round(k[index].x)
    plt.subplot(2, 2, index-4)
    plt.xticks(np.linspace(0, 300, 4))
    plt.yticks(np.linspace(-600, 600, 5))
    plt.xlim(0, 350)
    plt.xlabel("Speed (km/h)", fontsize=12)
    plt.ylabel('Force (kN)', fontsize=12)
    plt.grid(linestyle='-.', linewidth='0.5')
    if Notch_current == 1:
        plt.ylim(0, 550)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='1.5')
        for j in N_t:
            plt.plot(PWL_SPE, N_t[index], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, N_t['PWL_T1'], color='darkgreen', linewidth='1.5')
    elif Notch_current > 0:
        plt.ylim(0, 550)
        plt.plot(PWL_SPE, N_t['PWL_T' + str(Notch_current - 1)], color='darkgreen', linewidth='1.5')
        for j in N_t:
            plt.plot(PWL_SPE, N_t[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, N_t['PWL_T' + str(Notch_current)], color='darkgreen', linewidth='1.5')
    elif Notch_current == 0:
        plt.ylim(-300, 300)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='1.5')
    elif Notch_current == -1:
        plt.ylim(-800, 0)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='0.75')
        for j in N_b:
            plt.plot(PWL_SPE, -N_b[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, -N_b['PWL_B1'], color='darkgreen', linewidth='1.5')
    else:
        plt.ylim(-800, 0)
        plt.plot(PWL_SPE, -N_b['PWL_B' + str(-Notch_current - 1)], color='darkgreen', linewidth='1.5')
        for j in N_b:
            plt.plot(PWL_SPE, -N_b[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, -N_b['PWL_B' + str(-Notch_current)], color='darkgreen', linewidth='1.5')
    plt.plot(v_ave_i[index].x * 3.6, F_i[index].x / 1000, 'go')

# Figure 5
Fig5 = plt.figure(5)
for index in range(9, 13):
    Notch_current = round(k[index].x)
    plt.subplot(2, 2, index-8)
    plt.xticks(np.linspace(0, 300, 4))
    plt.yticks(np.linspace(-600, 600, 5))
    plt.xlim(0, 350)
    plt.xlabel("Speed (km/h)", fontsize=12)
    plt.ylabel('Force (kN)', fontsize=12)
    plt.grid(linestyle='-.', linewidth='0.5')
    if Notch_current == 1:
        plt.ylim(0, 550)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='1.5')
        for j in N_t:
            plt.plot(PWL_SPE, N_t[index], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, N_t['PWL_T1'], color='darkgreen', linewidth='1.5')
    elif Notch_current > 0:
        plt.ylim(0, 550)
        plt.plot(PWL_SPE, N_t['PWL_T' + str(Notch_current - 1)], color='darkgreen', linewidth='1.5')
        for j in N_t:
            plt.plot(PWL_SPE, N_t[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, N_t['PWL_T' + str(Notch_current)], color='darkgreen', linewidth='1.5')
    elif Notch_current == 0:
        plt.ylim(-300, 300)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='1.5')
    elif Notch_current == -1:
        plt.ylim(-800, 0)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='0.75')
        for j in N_b:
            plt.plot(PWL_SPE, -N_b[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, -N_b['PWL_B1'], color='darkgreen', linewidth='1.5')
    else:
        plt.ylim(-800, 0)
        plt.plot(PWL_SPE, -N_b['PWL_B' + str(-Notch_current - 1)], color='darkgreen', linewidth='1.5')
        for j in N_b:
            plt.plot(PWL_SPE, -N_b[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, -N_b['PWL_B' + str(-Notch_current)], color='darkgreen', linewidth='1.5')
    plt.plot(v_ave_i[index].x * 3.6, F_i[index].x / 1000, 'go')

# Figure 6
Fig6 = plt.figure(6)
for index in range(13, 17):
    Notch_current = round(k[index].x)
    plt.subplot(2, 2, index-12)
    plt.xticks(np.linspace(0, 300, 4))
    plt.yticks(np.linspace(-600, 600, 5))
    plt.xlim(0, 350)
    plt.xlabel("Speed (km/h)", fontsize=12)
    plt.ylabel('Force (kN)', fontsize=12)
    plt.grid(linestyle='-.', linewidth='0.5')
    if Notch_current == 1:
        plt.ylim(0, 550)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='1.5')
        for j in N_t:
            plt.plot(PWL_SPE, N_t[index], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, N_t['PWL_T1'], color='darkgreen', linewidth='1.5')
    elif Notch_current > 0:
        plt.ylim(0, 550)
        plt.plot(PWL_SPE, N_t['PWL_T' + str(Notch_current - 1)], color='darkgreen', linewidth='1.5')
        for j in N_t:
            plt.plot(PWL_SPE, N_t[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, N_t['PWL_T' + str(Notch_current)], color='darkgreen', linewidth='1.5')
    elif Notch_current == 0:
        plt.ylim(-300, 300)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='1.5')
    elif Notch_current == -1:
        plt.ylim(-800, 0)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='0.75')
        for j in N_b:
            plt.plot(PWL_SPE, -N_b[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, -N_b['PWL_B1'], color='darkgreen', linewidth='1.5')
    else:
        plt.ylim(-800, 0)
        plt.plot(PWL_SPE, -N_b['PWL_B' + str(-Notch_current - 1)], color='darkgreen', linewidth='1.5')
        for j in N_b:
            plt.plot(PWL_SPE, -N_b[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, -N_b['PWL_B' + str(-Notch_current)], color='darkgreen', linewidth='1.5')
    plt.plot(v_ave_i[index].x * 3.6, F_i[index].x / 1000, 'go')
# Figure 7
Fig7 = plt.figure(7)
for index in range(17, 21):
    Notch_current = round(k[index].x)
    plt.subplot(2, 2, index-16)
    plt.xticks(np.linspace(0, 300, 4))
    plt.yticks(np.linspace(-600, 600, 5))
    plt.xlim(0, 350)
    plt.xlabel("Speed (km/h)", fontsize=12)
    plt.ylabel('Force (kN)', fontsize=12)
    plt.grid(linestyle='-.', linewidth='0.5')
    if Notch_current == 1:
        plt.ylim(0, 550)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='1.5')
        for j in N_t:
            plt.plot(PWL_SPE, N_t[index], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, N_t['PWL_T1'], color='darkgreen', linewidth='1.5')
    elif Notch_current > 0:
        plt.ylim(0, 550)
        plt.plot(PWL_SPE, N_t['PWL_T' + str(Notch_current - 1)], color='darkgreen', linewidth='1.5')
        for j in N_t:
            plt.plot(PWL_SPE, N_t[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, N_t['PWL_T' + str(Notch_current)], color='darkgreen', linewidth='1.5')
    elif Notch_current == 0:
        plt.ylim(-300, 300)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='1.5')
    elif Notch_current == -1:
        plt.ylim(-800, 0)
        plt.plot([0, 350], [0, 0], color='darkgreen', linewidth='0.75')
        for j in N_b:
            plt.plot(PWL_SPE, -N_b[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, -N_b['PWL_B1'], color='darkgreen', linewidth='1.5')
    else:
        plt.ylim(-800, 0)
        plt.plot(PWL_SPE, -N_b['PWL_B' + str(-Notch_current - 1)], color='darkgreen', linewidth='1.5')
        for j in N_b:
            plt.plot(PWL_SPE, -N_b[j], 'k--', linewidth='0.75')
        plt.plot(PWL_SPE, -N_b['PWL_B' + str(-Notch_current)], color='darkgreen', linewidth='1.5')
    plt.plot(v_ave_i[index].x * 3.6, F_i[index].x / 1000, 'go')
