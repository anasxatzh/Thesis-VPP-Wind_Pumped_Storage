# VPP_PYPSA_2

import math
from statistics import median
from tkinter import W
from turtle import color
from matplotlib.pyplot import figure, subplot
import pypsa as ps
import numpy as np
import math
import pandas  as pd
import matplotlib.pyplot as plt
# from phes_final import Demand, Wind_Generated



# READ THE XLSX FILE
lesvos_behavior = pd.read_excel(r'C:\Users\user\Desktop\ΠΤΥΧΙΑΚΗ\Demand\Lesvos_Final\Merged_Sample_Lesvos.xlsx', index_col=0)
lesvos_behavior['date'] = pd.date_range(start = '01/01/2021',periods=len(lesvos_behavior), freq='H')
lesvos_behavior = lesvos_behavior.set_index('date')
electrical_load = lesvos_behavior['Load_Demand(MWh)']
conv_generation = lesvos_behavior['Conv_Generation(MWh)']
wind_abs = lesvos_behavior['Absorbed_Wind(MWh)']
w_rej = lesvos_behavior['Rejected_Wind(MWh)']


x = electrical_load.subtract(conv_generation)
new_d = x.subtract(wind_abs)


# phes_behaviour = pd.read_excel(r'C:\Users\user\Desktop\ΠΤΥΧΙΑΚΗ\Demand\Lesvos_Final\PHES_Lesvos_Sample.xlsx', index_col=0)
state = lesvos_behavior['STATE (0/1/2)'].to_frame()
p_flow = lesvos_behavior['Pump_Flow_Rate(m^3/s)'].to_frame()
t_flow = lesvos_behavior['Turb_Flow_Rate(m^3/s)'].to_frame()
energy_generated = lesvos_behavior['Energy_Generated(MWh)']
energy_stored = lesvos_behavior['Energy_Stored(MWh)']*2
inst_pump = lesvos_behavior['Installed_Pump_Power(MW)'].to_frame()
inst_turb = lesvos_behavior['Installed_Turb_Power(MW)'].to_frame()
water_lvl = lesvos_behavior['Water_Level(m)'].to_frame()
wind_gen = lesvos_behavior['Wind_Energy(MWh)'].to_frame()


g_import = []
w_import = []
for i in range(len(electrical_load)):
    if energy_stored[i] > 0:
        if new_d[i] <= 0 and w_rej[i] > 0 :
            w_import.append(w_rej[i])
            g_import.append(0)
        elif new_d[i] <=0 and w_rej[i] == 0:
            g_import.append(abs(new_d[i]))
            w_import.append(0)
    else:
        w_import.append(0)
        g_import.append(0)
# print(sum(w_import))
# print(sum(g_import))
# print(energy_stored.sum())


# HOURS PUMP (GRID & WIND), GEN, STAND
hours_pump = 0
hours_pump_wind = 0
hours_pump_grid = 0
hours_gen = 0
for i in range(len(electrical_load)):
    if energy_stored[i] > 0:
        hours_pump += 1
        if w_import[i] > 0:
            hours_pump_wind += 1
        else:
            hours_pump_grid += 1
    elif energy_generated[i] > 0:
        hours_gen += 1

tot_hours = 24
hours_stand = tot_hours - hours_gen - hours_pump

# print('gen', hours_gen)
# print('pump',hours_pump)
# print('stand',hours_stand)
# print('pump_grid',hours_pump_grid)
# print('pump_wind', hours_pump_wind)



# IF ENERGY_GENERATED LESS THAN ZERO => EQUAL TO ZERO
for i in range(len(energy_generated)):
    if energy_generated[i] < 0:
        energy_generated[i] = 0






installed_turb_max = inst_turb.max()
installed_turb_min = 0
installed_turb_nominal = 8


installed_pump_max = inst_pump.max()
installed_pump_min = 0
installed_pump_nominal = 9



# INSTALLED CONV POWER
conv_nominal = 81.33 # MW


# REJECTED WIND IMPORTED TO THE PUMPING SYSTEM
w_import_array = np.array(w_import)
w_import_nominal = w_import_array[np.nonzero(w_import_array)].mean()
w_import_min = 0
w_import_max = max(w_import)



# ARBITRAGE FOR THE HYBRID STATION
g_import_array = np.array(g_import)
g_import_nominal = g_import_array[np.nonzero(g_import_array)].mean()
g_import_min = min(g_import)
g_import_max = max(g_import)


pump_wind_rej =    [87, 85, 85, 84, 83, 81, 80, 78, 77, 76, 74, 74, 74, 73, 72, 71, 70, 70, 68, 67, 66, 65, 64, 63]
pump_grid_import = [13, 15, 15, 16, 17, 19, 20, 22, 23, 24, 26, 26, 26, 27, 28, 29, 30, 30, 32, 33, 34, 35, 36, 37]
w_abs_arr = np.array(wind_abs.tolist())
w_abs_nom = w_abs_arr[np.nonzero(w_abs_arr)].mean()
w_abs_min = wind_abs.min()
w_abs_max = wind_abs.max()


min_energy_stored = 15000
max_energy_stored = energy_stored.max()*hours_pump
energy_stored_arr = np.array(energy_stored.tolist())
med_energy_stored = energy_stored_arr[np.nonzero(energy_stored_arr)].mean() * hours_pump

# print(med_energy_stored)
# print(max_energy_stored)


# CONVERT TO DATAFRAME

g_import = pd.DataFrame(g_import)
w_import = pd.DataFrame(w_import)
g_import['date'] = pd.date_range(start = '01/01/2021',periods=len(lesvos_behavior), freq='H')
w_import['date'] = pd.date_range(start = '01/01/2021',periods=len(lesvos_behavior), freq='H')
g_import = g_import.set_index('date')
w_import = w_import.set_index('date')



energy_generated = energy_generated.to_frame()
electrical_load = electrical_load.to_frame()
energy_stored = energy_stored.to_frame()
conv_generation = conv_generation.to_frame()
wind_abs = wind_abs.to_frame()


wind_abs.loc[:,'p_max_pu'] = wind_abs/10.0

energy_generated.loc[:,'p_max_pu'] = energy_generated/8.0

electrical_load.loc[:,'power_mw'] = electrical_load/1.0

w_import.loc[:, 'p_max_pu'] = w_import/ w_import_nominal

conv_generation.loc[:, 'p_max_pu'] = conv_generation/ conv_nominal

conv_generation.loc[:, 'marg_cost'] = 128.5

wind_gen.loc[:, 'p_max_pu'] = wind_gen / 9.0

g_import.loc[:, 'p_set'] = g_import.iloc[:, 0]

g_import.loc[:, 'p_min_pu'] = 0






# # ECONOMIC EVALUATION

H_turb = 347.55
H_pump = 350.761
L_pen = 1900
D_pen = 1.3
A_pen = math.pi * (D_pen/2)**2
w_pen = 1979702
Q_t = 2.9
Q_p = 0.9
time_Q = 3600*hours_gen
# V_up = 47000 # m3
A_up = 100000 # m2
A_low = 2350 # m2

npv_list_100 = []
npv_list_150 = []
npv_rate = 0.35
npv_list_200 = []

# denom_npv_list = []
# npv_list = []


nominal_turb_power = [6.6, 7.5, 8.25, 9.075, 9.9, 10.8] # MW
nominal_pump_power = [7.6, 8.5, 9.5, 10.5, 11.4, 12.25]# MW


import numpy_financial as npf
# IC_t = 52000 * (installed_turb_nominal**(0.444)) * (H_turb**(-0.186)) # 997,840
# IC_p = 2 * 1814 * (installed_pump_nominal/ (H_pump**0.3))**(0.82)  # 750,000

for i in range(len(nominal_turb_power)):
    
    print('Installed_Pump is : {} MW'.format(nominal_pump_power[i]))
    print('')
    print('Installed_Gen is : {} MW'.format(nominal_turb_power[i]))
    print('')
    IC_t = 124730 * nominal_turb_power[i] 
    IC_p = 83330 * nominal_pump_power[i]

    V_up = nominal_turb_power[i] * 47000 / (8) # m3
    up_res_capacity = 7.6712766 * V_up * pow(10,-4) # m3 --> MWh


    # print('up_res (m3)', V_up)
    # print('up_res (MWh)', up_res_capacity)

# IC_up = 420 * V_up**(0.7) # 782,900
    IC_up = 21714 * up_res_capacity # euro/MWh


    IC_weld = 767 * L_pen #1,457,300
    IC_mat = 0.6* w_pen # 1,187,821
    IC_prot = 22 * A_pen  # 29.2
    IC_exc = 1.5 * L_pen * 5 * (math.pi * D_pen**(2) / 4) # 18,915
    IC_inst = 0.15 * IC_mat # 178,173 
    IC_pen = IC_weld + IC_mat + IC_prot + IC_exc + IC_inst # 2,842,238 



    IC_pilot = IC_t + IC_p + IC_pen + IC_up
    IC_land = 50 * 80000 # 4,000,000
    IC_cons = 0.15 * IC_pilot # 805,000
    IC_transfer = 0.024 * IC_pilot # 128,950
    IC_grid = 0.04 * IC_pilot # 215,000
    IC_control = 0.016 * IC_pilot # 85,968
    IC_road = 100000 
    IC_other = IC_land + IC_cons + IC_transfer + IC_grid + IC_control + IC_road # 5,235,000

    IC_initial = IC_other + IC_pilot # 10,607,897

# print('penstock', IC_pen)
# print('turbine', IC_t)
# print('pump', IC_p)
# print('reservoir', IC_up)

# print('pilot', IC_pilot)


# print('land', IC_land)
# print('grid', IC_grid)
# print('control', IC_control)
# print('transfer', IC_transfer)
# print('consulting', IC_cons)
# print('other', IC_other)
# print('Initial_Cost', IC_initial)


# E_g = energy_generated.iloc[:, 0].sum()
# E_rej = w_import.iloc[:, 0].sum()
# E_grid = g_import.iloc[:, 0].sum()

    t_gen = 1639 # h
    t_pump = 2176 # h

    E_g = t_gen * nominal_turb_power[i] # MWh
    E_rej = 0.648 * nominal_pump_power[i] * t_pump # MWh
    E_grid = 0.352 * nominal_pump_power[i] * t_pump # MWh
    E_abs = E_rej + E_grid # MWh
    e_rej = 0.03
    e_abs = 0.03
    e_import = 0.03
    e_g = 0.03


    d_rate = 0.07 # discount_rate
    n_payback_loan = 12 # years to payback the loan
    n_lifetime = 25 # lifetime of the investment
    inflation_rate = 0.0355 
    loan_interest = 0.06 
    tax_rate = 0.41

    c_import_grid = 75 # cost of purchasing energy from the grid
    c_rej_wind = 40 # cost of purchasing energy from the rej wind power
    c_abs_wind = 100 # revenues from the absorbed wind energy


# revenues from the guaranteed phes energy
    # c_g_1 = 150
    # c_g_1 = 100
    c_g_1 = 200 

    print('Revenues from the guaranteed energy are : {} €/MWh'.format(c_g_1))
    print('')
    var_r = ( ((1 + e_g) / (1+d_rate))**(n_lifetime) - 1 ) / (e_g - d_rate)
    tso_rate = 0.03 # percentage of revenues going to the TSO

    # TOTAL INCOMES (IF)
    IF_total = E_g * c_g_1 * (1+e_g) * var_r + c_abs_wind * E_abs * (1+e_abs) * var_r

    # print('total_incomes', IF_total)

# TOTAL EXPENSES (OF)
    a_1 = 0.45
    b_1 = 0.2
    c_1 = 0.35

    # a_1 = 0.4
    # b_1 = 0.2
    # c_1 = 0.4

    # a_1 = 0.2
    # b_1 = 0.2
    # c_1 = 0.6
    print('The study case scheme is : {}% equity, {}% loan amortization and {}% subsidy'.format(a_1*100, b_1*100, c_1*100))
    print('')


# loan
    t = b_1 * IC_initial
    s = t / n_payback_loan
    j = [i for i in range(1, n_lifetime + 1)]
    total_denominator = 0
    denom_list = []
    numer_list = []

    for k,c in enumerate(j):
        total_denominator = (1+d_rate) ** c
        denom_list.append(total_denominator)

    dc_list = [t,t-s,t-2*s,t-3*s,t-4*s,t-5*s,t-6*s,t-7*s,t-8*s,t-9*s,t-10*s,t-11*s,t-12*s,t-13*s,t-14*s,t-15*s,t-16*s,t-17*s,t-18*s,t-19*s,t-20*s,t-21*s,t-22*s,t-23*s,t-24*s]

    for i in range(len(dc_list)):
        dc_list[i] *= loan_interest
        dc_list[i] += s
        numer_list.append(dc_list[i])
    loan_list = []
    for i in range(len(denom_list)):
        final_var = numer_list[i] / denom_list[i]
        loan_list.append(final_var)

    L_n = sum(loan_list)

    print('The total amount of loans are : {} €'.format(L_n))
    print('')

# rejected wind energy purchase
    W_n = E_rej * c_rej_wind * (1+e_rej) * var_r

    print('Purchasing rejected wind power costs : {} €'.format(W_n))
    print('')

    # import energy from the grid
    C_n = E_grid * c_import_grid * (1+e_import) * var_r
    print('Purchasing energy from the grid costs : {} €'.format(C_n))
    print('')

# operation and maintenance costs
    O_M_n = 0.02 * IC_pilot
    print('The O&M costs are : {} € '.format(O_M_n))
    print('')

    # TSO
    F_n = tso_rate * IF_total
    print('The total amount of money paid to the TSO is {} €'.format(F_n))
    print('')

# taxes
    my_list_1 = []
    my_list_2 = []
    my_list_3 = []
    fin_list = []
    denom_tax = 0
    denom_tax_list = []
    denom_tax_list_2 = []
    var_minus_list = []
    denom_tax_2 = 0
    W_C = W_n + C_n
    IF_list = [(IF_total / n_lifetime)for i in range(n_lifetime)]
    O_M_n_list = [(O_M_n / n_lifetime)for i in range(n_lifetime)]
    W_C_list = [(W_C / n_lifetime)for i in range(n_lifetime)]

    for k, c in enumerate(j):
        denom_tax = (1+d_rate) ** c
        denom_tax_list.append(denom_tax)
    for i in range(len(denom_tax_list)):
        var_minus = IF_list[i] - O_M_n_list[i] - W_C_list[i]
        var_minus_list.append(var_minus)
        var_minus_list[i] *= tax_rate
        var_my_list_1 = var_minus_list[i] / denom_tax_list[i] 
        my_list_1.append(var_my_list_1)
    sum_list_1 = sum(my_list_1)

    for k, c in enumerate(j):
        denom_tax_2 = (1+d_rate) ** c
        denom_tax_list_2.append(denom_tax_2)

    for i in range(len(dc_list)):
        dc_list[i] *= tax_rate
        my_list_2.append(dc_list[i])
        my_list_3.append(my_list_2[i] / denom_tax_list_2[i])

    sum_list_3 = tax_rate * sum(my_list_3)
    T_n = sum_list_1 - sum_list_3

    print('The total taxes are {} €'.format(T_n))
    print('')
    # total expenses
    OF_total = W_n + C_n + O_M_n + L_n + T_n + F_n

    print('THE TOTAL EXPENSES ARE : {} €'.format(OF_total))
    print('')

# Total Revenues
    R_total = IF_total - OF_total - a_1*IC_initial

    print('THE TOTAL REVENUES ARE : {} €'.format(R_total))
    print('')
    R_t = IF_total - OF_total


    equity = a_1 * IC_initial
    loan_amortization = b_1 * IC_initial
    subsidy = c_1 * IC_initial
    print('The initial cost is {}, the amount of equity is {}, loan {} and subsidy {}'.format(IC_initial, equity, loan_amortization, subsidy))
    print('')


# Net Present Value

    R_t_list = [(R_t / n_lifetime) for i in range(n_lifetime)]
    denom_npv_list = []
    npv_list = []

    denominator = 0
    for k,c in enumerate(j):
        denominator = (1+d_rate) ** c
        denom_npv_list.append(denominator)

    for i in range(len(denom_npv_list)):
        npv_value = R_t_list[i] / denom_npv_list[i] 
        npv_list.append(npv_value)
        
    npv_final_value = (sum(npv_list) - equity) * (1 - npv_rate)
     
    # npv_list_100.append(npv_final_value)
    npv_list_200.append(npv_final_value)
    # npv_list_150.append(npv_final_value)

    print('THE NPV VALUE IS : {} €'.format(npv_final_value))
    print('')


# RES PENETRATION LEVEL:
    en_absorbed_pump = E_abs
    en_absorbed_wind = 21105 # MWh
    en_generated_turbine = E_g
    annual_load_demand = 327000 # MWh
    en_generated_rest_renewables = 0 # MWh

    res_pen_level = ((en_absorbed_wind + en_generated_turbine + en_generated_rest_renewables - en_absorbed_pump ) / annual_load_demand ) * 100

    print('The hybrid stations penetration level is: {} %'.format(res_pen_level))
    print('')

# Payback period in years

    OF_list = [(OF_total / n_lifetime) for i in range(n_lifetime)]
    cash_flows = 0
    cash_flow_list = []
    pay_back_list = []
    for i in range(len(OF_list)):
        cash_flows += IF_list[i] - OF_list[i]
        cash_flow_list.append(cash_flows)
        pay_back_formula = cash_flow_list[i] - equity
        pay_back_list.append(pay_back_formula)


    for i in range(len(OF_list)):
        if (pay_back_list[i] <= 0 and pay_back_list[i+1] > 0) :
            payback_period = i + ( abs(pay_back_list[i]) / (cash_flow_list[i+1] - cash_flow_list[i]))
    
    print('The payback period is {} years'.format(payback_period))
    print('')


# Internal rate of return 

    irr_list = []
    for i in range(len(OF_list)):
        cash_flows_irr = IF_list[i] - OF_list[i]
        irr_list.append(cash_flows_irr)
    irr_list.insert(0, -IC_initial)


    final_irr = npf.irr(irr_list)

    print('The internal rate of return is {}%'.format(round(final_irr * 100,2)))
    print('')
    print('')


# print(npv_list_100)
# print(npv_list_150)
# print(npv_list_200)




# npv_list_100_final = [-2116661.8069878276, -857909.822163023, 293948.465327736, 1519833.2615128811, 2704558.7376433127, 3942731.062440753]
# npv_list_150_final = [8113596.528295786, 9712612.05048831, 11173768.895473113, 12729583.861026665, 14233941.262650445, 15807228.002878085]
# npv_list_200_final = [11219493.266417742, 13242040.161990521, 15056139.81812556, 17000191.87594434, 18892786.369833373, 20889604.48344129] # 2o

# for i in range(len(npv_list_100_final)):
    # npv_list_100_final[i] = npv_list_100_final[i] * pow(10,-6)
    # npv_list_150_final[i] = npv_list_150_final[i] * pow(10,-6) * 0.625
    # npv_list_200_final[i] = npv_list_200_final[i] * pow(10,-6) * 0.75

# npv_list_1_scenario = [5.070997830184866, 6.070382531555193, 6.983605559670696, 7.955989913141666, 8.896213289156528, 9.879517501798802]
# npv_list_2_scenario = [11728589.549254885, 13768069.243236233, 15597560.763461744, 17558031.392280042, 19466531.9776686, 21480026.64993508]
# npv_sc_3 = [-2065342.7658887357, -484656.1981350668, 927793.7159630768, 2443404.178416349, 3920930.197814904, 5482574.544041213]

# npv_list_100_final = [-2116661.8069878276, -857909.822163023, 293948.465327736, 1519833.2615128811, 2704558.7376433127, 3942731.062440753]
# npv_list_150_final = [8113596.528295786, 9712612.05048831, 11173768.895473113, 12729583.861026665, 14233941.262650445, 15807228.002878085]
# npv_list_200_final = [8622692.81113293, 10238641.131734021, 11715189.840809297, 13287423.377362369, 14807686.87048567, 16397650.169371877] # 2o

# for i in range(len(npv_list_100_final)):
#     npv_list_100_final[i] = npv_list_100_final[i] * pow(10,-6)
#     npv_list_150_final[i] = npv_list_150_final[i] * pow(10,-6)
#     npv_list_200_final[i] = npv_list_200_final[i] * pow(10,-6)

# npv_list_100_final = [-592359.32103475, 396864.0452832598, 1335767.5741226487, 2322741.1664162315, 3263894.469280055, 4230207.690883191]
# npv_list_150_final = [2513537.417087203, 3926292.156785488, 5218138.496775093, 6593349.181333926, 7922739.576462988, 9312584.171446387]
# npv_list_200_final = [5619434.155209158, 7455720.2682877, 9100509.41942754, 10863957.196251601, 12581584.683645915, 14394960.652009591]
# for i in range(len(npv_list_200)):
#     npv_list_200_final[i] = npv_list_200_final[i] * pow(10,-6)
#     npv_list_100_final[i] = npv_list_100_final[i] * pow(10,-6)
#     npv_list_150_final[i] = npv_list_150_final[i] * pow(10,-6)










# X_axis = np.arange(1, 13, 0.5)
# width = 0.25
# width_2 = -0.04
# fig = plt.figure(figsize = (8,6))

# plt.bar(X_axis, pump_grid_import, color = 'r', label = 'Ενεργειακές εισαγωγές', width = width)
# plt.bar(X_axis, pump_wind_rej, bottom = pump_grid_import, color = 'b', label = 'Απορριπτόμενη αιολική ενέργεια', width = width)
# plt.xlabel('Εγκατεστημένη ισχύς Υ/Σ')
# plt.ylabel('Σύνολική ενέργεια άντλησης')
# plt.legend(loc = 'upper right')
# plt.show()




# FOR THE THREE GUAR_ENERGY PRICES

# cg_100_1 = [3255004.8636129918, 4019069.560340953, 4739408.682333435, 5498334.299970832, 6223812.501053885, 6971153.489504678]

# cg_150_1 = [5273837.743392261, 6313197.832817402, 7262949.782057524, 8274229.509667332, 9252061.82072279, 10274698.201870756]
 
# cg_200_1 = [7292670.6231715325, 8607326.10529384, 9786490.881781615, 11050124.71936382, 12280311.140391693, 13578242.914236838]

# for i in range(len(cg_100_1)):
#     cg_100_1[i] = cg_100_1[i] * pow(10,-6) 
#     cg_150_1[i] = cg_150_1[i] * pow(10,-6) 
#     cg_200_1[i] = cg_200_1[i] * pow(10,-6) 

# y1 = cg_100_1
# x1 = nominal_turb_power
# label_1 = 'Cg = 100 €/MWh'
# color_1 = 'blue'

# y2 = cg_150_1
# x2 = nominal_turb_power
# label_2 = 'Cg = 150 €/MWh'
# color_2 = 'red'

# y3 = cg_200_1
# x3 = nominal_turb_power
# label_3 = 'Cg = 200 €/MWh'
# color_3 = 'black'

# plt.figure(figsize=(12,6))
# plt.plot(x1,y1, label = label_1, color = color_1)
# plt.plot(x2,y2, label = label_2, color = color_2)
# plt.plot(x3,y3, label = label_3, color = color_3)

# plt.xlabel('Εγκατεστημένη Ισχύς Υ/Σ (MW)')
# plt.ylabel('Καθαρά παρούσα αξία (εκατομμύρια €)')
# plt.title('Μεταβολή της καθαράς παρούσας αξίας')
# plt.tight_layout()
# plt.legend()
# plt.show()





# # FOR THE 3 ECONOMIC SCHEMA!!

# npv_list_1_scenario = [5273837.743392261, 6313197.832817402, 7262949.782057524, 8274229.509667332, 9252061.82072279, 10274698.201870756] # 1o
# npv_list_2_scenario = [6928400.662612978, 8022792.346865962, 9022567.85440012, 10087207.93775837, 11116735.046187267, 12193570.242975576]
# npv_sc_3 = [-2065342.7658887357, -484656.1981350668, 927793.7159630768, 2443404.178416349, 3920930.197814904, 5482574.544041213]

# for i in range(len(npv_list_1_scenario)):
#     npv_list_2_scenario[i] = npv_list_2_scenario[i] * pow(10,-6) 
#     npv_list_1_scenario[i] = npv_list_1_scenario[i] * pow(10,-6) 
#     npv_sc_3[i] = npv_sc_3[i] * pow(10,-6) 

# y1 = npv_list_1_scenario
# x1 = nominal_turb_power
# label_1 = '45% ίδια κεφάλαια, 35% κρατική επιχορήγηση, 20% λήψη δανείου'
# color_1 = 'blue'

# y2 = npv_list_2_scenario
# x2 = nominal_turb_power
# label_2 = '40% ίδια κεφάλαια, 40% κρατική επιχορήγηση, 20% λήψη δανείου'
# color_2 = 'red'

# y3 = npv_sc_3
# x3 = nominal_turb_power
# label_3 = '70% ίδια κεφάλαια, 30% λήψη δανείου'
# color_3 = 'black'

# plt.figure(figsize=(12,6))
# plt.plot(x1,y1, label = label_1, color = color_1)
# plt.plot(x2,y2, label = label_2, color = color_2)
# plt.plot(x3,y3, label = label_3, color = color_3)

# plt.xlabel('Εγκατεστημένη Ισχύς Υ/Σ (MW)')
# plt.ylabel('Καθαρά παρούσα αξία (εκατομμύρια €)')
# plt.title('Καθαρά παρούσα αξία για τα 3 σενάρια χρηματοδότησης')
# plt.tight_layout()
# plt.legend()
# plt.show()

































































