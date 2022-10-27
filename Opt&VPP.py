from itertools import count
from logging.config import DEFAULT_LOGGING_CONFIG_PORT
from mimetypes import init
import pypsa as ps
import pandas as pd
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt
import time 
from Wind_Gen_Abs_Rej import wind_speed_func



lesvos_behavior = pd.read_excel(r'C:\Users\user\Desktop\ΠΤΥΧΙΑΚΗ\Demand\Lesvos_Final\les_beh_final_2.xlsx', index_col=0)
lesvos_behavior['date'] = pd.date_range(start = '01/01/2021',periods=len(lesvos_behavior), freq='H')
lesvos_behavior = lesvos_behavior.set_index('date')


# lesvos_behavior = lesvos_behavior.sample(n = 200)

electrical_load = lesvos_behavior['Load_Demand(MWh)']
conv_generation = lesvos_behavior['Conv_Generation(MWh)']
wind_abs = lesvos_behavior['Absorbed_Wind(MWh)']
w_rej = lesvos_behavior['Rejected_Wind(MWh)']
w_generated = lesvos_behavior['Wind_Energy(MWh)'].to_frame()
s_cor = 3.37 # time_looping
x = electrical_load.subtract(conv_generation)
new_d = x.subtract(wind_abs)


# les_beh = lesvos_behavior.to_excel('les_beh_final_2.xlsx')


state = lesvos_behavior['STATE (0/1/2)'].to_frame()
p_flow = lesvos_behavior['Pump_Flow_Rate(m^3/s)'].to_frame()
t_flow = lesvos_behavior['Turb_Flow_Rate(m^3/s)'].to_frame()
energy_generated = lesvos_behavior['Energy_Generated(MWh)']
energy_stored = lesvos_behavior['Energy_Stored(MWh)']*2
inst_pump = lesvos_behavior['Installed_Pump_Power(MW)'].to_frame()
inst_turb = lesvos_behavior['Installed_Turb_Power(MW)'].to_frame()
s_cor_2 = 2.44
water_lvl = lesvos_behavior['Water_Level(m)'].to_frame()



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







for i in range(len(energy_generated)):
    x = electrical_load[i] - wind_abs[i] - conv_generation[i]
    if x > 0 and energy_generated[i] == 0:
        energy_generated[i] = x

conv_el = []
for i in range(len(conv_generation)):
    conv_el.append(conv_generation[i] - g_import[i])

conv_el_series = pd.Series(conv_el)

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

tot_hours = 730
hours_stand = tot_hours - hours_gen - hours_pump


g_import = pd.DataFrame(g_import)
w_import = pd.DataFrame(w_import)

tot_energy = energy_stored.add(energy_generated)
tot_energy = tot_energy.to_frame()
tot_energy.loc[:, 'p_max_pu'] = tot_energy/9.0
tot_energy_pu = tot_energy.iloc[:,1]
tot_energy.loc[:, 'stored'] = energy_stored
tot_energy.loc[:, 'gen'] = energy_generated


energy_stored = energy_stored.to_frame()
conv_generation = conv_generation.to_frame()
# wind_abs = wind_abs.to_frame()
energy_generated = energy_generated.to_frame()
# w_rej = w_rej.to_frame()
lower2upper_energy = energy_stored.iloc[:, 0]

conv_nominal = 81.33
conv_generation.loc[:, 'p_max_pu'] = conv_generation/ conv_nominal

conv_gen_p_set = conv_generation.iloc[:, 0]


conv_generation.insert(2, 'conv_elec', conv_el, True)
g_import_list = g_import.iloc[:, 0].tolist()
conv_generation.insert(3, 'g_import', g_import_list, True)



energy_generated_series = energy_generated.iloc[:, 0]
energy_generated.loc[:,'p_max_pu'] = energy_generated/8.0
energy_gen_max_pu = energy_generated.loc[:,'p_max_pu']
obj = conv_nominal / s_cor_2
energy_gen_nom_max = (energy_gen_max_pu * 8.0).max()

tot_hours = 8760
w_generated.loc[:, 'p_max_pu'] = w_generated / 9.0
w_gen_p_set = w_generated.iloc[:, 0]
w_gen_max_pu = w_generated.loc[:, 'p_max_pu']
w_gen_max_nom = w_generated['Wind_Energy(MWh)'].max()
t_t = w_gen_max_nom / s_cor
c_gen = conv_generation.iloc[:, 0]
gen_value = 36.55
pump_value = 67.04
c_elec = conv_generation.iloc[:, 2]
c_g_import = conv_generation.iloc[:, 3]


efficiency_w_rej = []
efficiency_w_abs = []
efficiency_conv_elec = []
efficiency_g_import = []
efficiecny_energy_stored = []
efficiecny_energy_gen = []

total_energy = tot_energy.iloc[:,0]
# en_stored & en_gen
for i in range(len(total_energy)):
    if total_energy[i] != 0 :
        if lower2upper_energy[i] > 0:

            x_stored = lower2upper_energy[i] / total_energy[i]
            efficiecny_energy_stored.append(x_stored)
            x_generated = 0
            efficiecny_energy_gen.append(x_generated)
        else:
            x_stored = 0
            x_generated = energy_generated_series[i] / total_energy[i]
            efficiecny_energy_stored.append(x_stored)
            efficiecny_energy_gen.append(x_generated)   
    else:
        x_stored = 0
        x_generated = 0
        efficiecny_energy_gen.append(x_generated)
        efficiecny_energy_stored.append(x_stored)


# c_elec & c_import
for i in range(len(conv_generation)):
    if c_gen[i] != 0:
        x_elec = c_elec[i] / c_gen[i] 
        x_import = c_g_import[i] / c_gen[i]
        efficiency_conv_elec.append(x_elec)
        efficiency_g_import.append(x_import)        
    
    else:
        x_elec = 0
        x_import = 0
        efficiency_conv_elec.append(x_elec)
        efficiency_g_import.append(x_import)


# w_abs & w_rej
for i in range(len(w_gen_p_set)):
    if w_gen_p_set[i] != 0: 
        x_abs = wind_abs[i] / w_gen_p_set[i]     
        efficiency_w_abs.append(x_abs)
        x_rej = w_rej[i] / w_gen_p_set[i]
        efficiency_w_rej.append(x_rej)   
    else:
        x_abs = 0
        x_rej = 0
        efficiency_w_abs.append(x_abs)
        efficiency_w_rej.append(x_rej) 


conv_gen = {
    'conv_nominal' : conv_nominal,
    'conv_min_pu' : 0,
    'conv_max_pu' : conv_generation['p_max_pu'],
    'conv_p_set' : conv_gen_p_set,
}
phes_gen = {
    'nom' : 8,
    'cap_cost' : 124730,
    'max_pu' : energy_gen_max_pu,
    'gen_eff' : 0.81,
    'p_set' : energy_generated_series
    
}

wind_gen = {
    'w_nom' : 9,
    'w_max_pu' : w_gen_max_pu ,
    'w_min_pu' : 0,
    'wind_set' : w_gen_p_set,
    'cap_cost' : 0,
    'w_absorbed' : wind_abs ,
    'w_rejected' : w_rej ,
    'max_wind' : w_gen_max_nom
}




override_component_attrs = ps.descriptors.Dict(
    {k: v.copy() for k, v in ps.components.component_attrs.items()}
)
override_component_attrs["Link"].loc["bus2"] = [
    "string",
    np.nan,
    np.nan,
    "2nd bus",
    "Input (optional)",
]
override_component_attrs["Link"].loc["efficiency2"] = [
    "static or series",
    "per unit",
    1.0,
    "2nd bus efficiency",
    "Input (optional)",
]
override_component_attrs["Link"].loc["p2"] = [
    "series",
    "MW",
    0.0,
    "2nd bus output",
    "Output",
]
override_component_attrs["Link"].loc["marginal_cost2"] = [
    "static or series",
    "currency/MWh",
    0.0,
    "2nd bus marginal_cost",
    "Input (optional)",
]
override_component_attrs["Link"].loc["marginal_cost1"] = [
    "static or series",
    "currency/MWh",
    0.0,
    "2nd bus marginal_cost",
    "Input (optional)",
]

n = ps.Network(override_component_attrs=override_component_attrs)
n.set_snapshots(lesvos_behavior.index)
n.snapshot_weightings = pd.Series(data = 1, index = n.snapshots, dtype = 'object')


n.add('Bus', 'el_bus')
n.add('Bus', 'wind_bus')
n.add('Bus', 'pump_bus')
n.add('Bus', 'gen_bus')
n.add('Bus', 'conv_bus')


n.add('Load',
      'electricity_load',
      bus = 'el_bus',
      p_set = electrical_load
    )




n.add('Generator',
      'conv_generator',
      bus = 'conv_bus',
      p_nom = conv_gen['conv_nominal'],
      p_min_pu = conv_gen['conv_min_pu'],
      p_max_pu = conv_gen['conv_max_pu'],
    #   p_set = conv_gen['conv_p_set'],
      capital_cost = 0,
    #   marginal_cost = m_cost_conv # euro/MWh
    )


n.add('Generator',
     'phes_generator',
     bus = 'gen_bus',
     p_nom_extendable = True,
     p_nom_min = 0,
     p_nom_max = 15,
    #  p_max_pu = phes_gen['max_pu'],
     capital_cost = phes_gen['cap_cost'],
    #  marginal_cost = -200.0 , # euro/MWh
    )

n.add('Generator',
     'wind_generator',
     bus = 'wind_bus',
     p_nom = wind_gen['max_wind'],
    #  p_set = wind_gen['wind_set'] ,
     p_max_pu = wind_gen['w_max_pu'],
     p_min_pu = wind_gen['w_min_pu'],
     capital_cost = wind_gen['cap_cost'],
    #  marginal_cost = -60 # euro/MWh
)

n.add('Link',
     'w_abs&w_rej',
    #  p_nom = wind_gen['max_wind'],
     bus0 = 'wind_bus',
     bus1 = 'el_bus',
     bus2 = 'pump_bus',
     marginal_cost1 = -100, # euro/MWh
     marginal_cost2 = 40, # euro/MWh
     p_nom_extendable = True,
    #  p_nom_min = 0,
    #  p_nom_max = wind_gen['max_wind'],
     capital_cost = 0,
     efficiency = efficiency_w_abs,
     efficiency2 = efficiency_w_rej,
)


n.add('Link',
     'g_import',
    #  p_nom = conv_gen['conv_nominal'],
     bus0 = 'conv_bus',
     bus1 = 'el_bus',
     bus2 = 'pump_bus',
    p_nom_extendable = True,
    p_nom_min = 0,
    p_nom_max = conv_gen['conv_nominal'],
     efficiency = efficiency_conv_elec,
     efficiency2 = efficiency_g_import,
     capital_cost = 0,
    #  marginal_cost1 = 100, # euro/MWh
    #  marginal_cost2 = 45 # euro/MWh
)

pump_en = lower2upper_energy.add(phes_gen['p_set'])
pump_en_max = lower2upper_energy.add(phes_gen['p_set']).max()




n.add('Generator',
     'pump',
     bus = 'pump_bus',
     sign = -1,
     p_nom_extendable = True,
     p_nom_min = 0,
    #  p_nom_max = 8,
    #  p_max_pu = tot_energy_pu,
    #  capital_cost = 83330, # euro/MW,
)


n.add('Link',
      'lower_to_upper',
      bus0 = 'pump_bus',
      bus1 = 'gen_bus',
      p_nom_extendable = True,
      p_nom_min = 0,
      p_nom_max = pump_en_max,
      capital_cost = 0,
      efficiency = efficiecny_energy_stored,
    #   efficiency2 = efficiecny_energy_gen,
    #   marginal_cost1 = 0,
    #   marginal_cost2 = -200 # euro/MWh
    )

up2low_max = energy_generated_series.max()

n.add('Link',
      'upper_to_lower',
      bus0 = 'gen_bus',
      bus1 = 'el_bus',
      bus2 = 'pump_bus',
      p_nom_extendable = True,
      p_nom_min = 0,
      p_nom_max = up2low_max,
    #   capital_cost = 5235000,
      efficiency = efficiecny_energy_gen,
      efficiency2 =  efficiecny_energy_stored
)



max_stored_up = energy_stored.max()
n.add('Store',
     'upper_res',
     bus = 'gen_bus',
     e_nom_extendable = True,
     e_nom_min = 0,
    #  e_nom_max = 100,
     e_cyclic = True,
    #  capital_cost = -21714 # euro/MWh
)

n.generators.build_year = n.generators.build_year.astype(np.int32)
n.generators.min_up_time = n.generators.min_up_time.astype(np.int32)
n.generators.min_down_time = n.generators.min_down_time.astype(np.int32)
n.generators.up_time_before = n.generators.up_time_before.astype(np.int32)
n.generators.down_time_before = n.generators.down_time_before.astype(np.int32)

n.stores.build_year = n.stores.build_year.astype(np.int32)
n.links.build_year = n.links.build_year.astype(np.int32)

solv_name = 'gurobi'

n.lopf(n.snapshots, solver_name = solv_name, pyomo = False)
# print('objective' , n.objective*y_factor)

# print('Optimal_installed_phes', n.generators.p_nom_opt['phes_generator'] * x_factor)
# print('Optimal_installed_pump',n.generators.p_nom_opt['pump'] * x_factor)
v_up_res = n.stores.e_nom_opt * pow(10, 4) / 15.34
# print(n.links.p_nom_opt)
gen_installed_opt = n.generators.p_nom_opt['phes_generator']
pump_installed_opt = n.generators.p_nom_opt['pump']
opt_value = n.objective
# print('Optimal_upper_res_capacity: {} m3'.format(v_up_res))
print(n.links_t.p0)
print(n.links_t.p1)
print(n.links_t.p2)

# print(n.stores_t.e)
a = pd.DataFrame({attr: n.stores_t[attr]["upper_res"] for attr in ["p", "e"]})
load_demand_series = lesvos_behavior['Load_Demand(MWh)']
w_generated_series = n.links_t.p0['w_abs&w_rej'] # w_energy_generated
loop_frames_1 = 27
loop_frames_2 = 17
loop_frames_3 = 37
loop_frames_final = 44
stand_value = tot_hours / len(lesvos_behavior)
obj_value = opt_value*obj
w_abs_series = n.links_t.p1['w_abs&w_rej']  # w_energy_absorbed
w_rej_series = n.links_t.p2['w_abs&w_rej'] # w_energy_rejected
gen_installed = gen_installed_opt*t_t
conv_gen_series = n.links_t.p0['g_import'] # conv_energy_generated
conv_load = n.links_t.p1['g_import']
g_import_series = n.links_t.p2['g_import'] # conv_energy_imported
pump_installed = pump_installed_opt*t_t
en_stored_series = n.links_t.p1['lower_to_upper']
en_generated_series = n.links_t.p1['upper_to_lower']

lesvos_frame = {
    'Load_Demand' : load_demand_series,
    'Wind_Energy_Generated' : w_generated_series,
    'Wind_Energy_Rejected' : w_rej_series,
    'Wind_Energy_Absorbed' :w_abs_series ,
    'Conv_Energy_Generated' : conv_gen_series,
    'Conv_Energy_To_Load' :conv_load ,
    'Conv_Energy_Imported' : g_import_series,
    'Energy_Generated' :en_generated_series ,
    'Energy_Stored' : en_stored_series    
}
df_lesvos = pd.DataFrame(lesvos_frame)
# print(df_lesvos.head())
frames_1 = []
frames_2 = []
frames_3 = []
frames_final = []
for i in range(loop_frames_1):
    frames_1.append(df_lesvos)

df_lesvos_1 = pd.concat(frames_1)


for i in range(loop_frames_2):
    frames_2.append(df_lesvos)
df_lesvos_2 = pd.concat(frames_2)


for i in range(loop_frames_3):
    frames_3.append(df_lesvos)

df_lesvos_3 = pd.concat(frames_3)

y = -1
tot_gen = y *df_lesvos_2['Energy_Generated'].sum()
tot_stored = y *df_lesvos_1['Energy_Stored'].sum()


for i in range(loop_frames_final):
    frames_final.append(df_lesvos)

final_df_lesvos = pd.concat(frames_final)

index_num = []
for i in range(8800):
    index_num.append(i)

final_df_lesvos['index_num'] = index_num
final_df_lesvos = final_df_lesvos.set_index('index_num')

rows_drop = []
for j in range(8760, 8800):
    rows_drop.append(j)

final_df_lesvos.drop(rows_drop, axis = 0, inplace = True)


final_df_lesvos['date'] = pd.date_range(start = '01/01/2021',periods=8760, freq='H')
final_df_lesvos = final_df_lesvos.set_index('date')


energy_stored_grid = y * df_lesvos_3['Conv_Energy_Imported'].sum()
energy_stored_wind = tot_stored - energy_stored_grid


final_df_lesvos['Wind_Energy_Rejected'] = final_df_lesvos['Wind_Energy_Rejected'] * y
final_df_lesvos['Wind_Energy_Absorbed'] = final_df_lesvos['Wind_Energy_Absorbed'] * y
final_df_lesvos['Conv_Energy_To_Load'] = final_df_lesvos['Conv_Energy_To_Load'] * y
final_df_lesvos['Conv_Energy_Imported'] = final_df_lesvos['Conv_Energy_Imported'] * y
final_df_lesvos['Energy_Generated'] = final_df_lesvos['Energy_Generated'] * y
final_df_lesvos['Energy_Stored'] = final_df_lesvos['Energy_Stored'] * y

# final_df_excel = final_df_lesvos.to_excel('PyPSA_final_outputs.xlsx', sheet_name='Outputs')


# print(len(final_df_lesvos))

# print(final_df_lesvos.head())


# HYBRID POWER STATION

# sleeping_time = 15 # sec
# small_sleeping_time = 4 # sec
sleeping_time = 0
small_sleeping_time = 0


print('')
print('IN THIS PARTICULAR PROJECT A HYBRID POWER STATION IS IMPLEMENTED IN THE NON-INTERCONNECTED GRID OF LESVOS ISLAND')
print('')

cost_of_grid_import = 75 # euro/MWh
cost_of_rej_wind = 40 # euro/Mwh
cost_of_abs_wind = 100 # euro/MWh
cost_of_guar_energy = 150 # euro/MWh

initial_budget = 0


x = -1

date_time_vpp = pd.date_range(start = '01/01/2021',periods=len(lesvos_behavior), freq='H')

fuel_cons_cost = 270 #kg/MWh
tot_fuel_cons = 0

wind_speed = wind_speed_func()

wind_speed_list_200 = wind_speed[0:200]

energy_stored_upper_res = 10 # MWh

count_gen = 0
count_pump = 0
count_stand = 0
time.sleep(small_sleeping_time)
for i in range(len(df_lesvos)):
    
    print('The date is : {} '.format(date_time_vpp[i]))
    time.sleep(small_sleeping_time)
    
    print('The Load Demand is equal to {} MWh.'.format(df_lesvos['Load_Demand'][i]))
    time.sleep(small_sleeping_time)
    
    print('The energy stored in the upper reservoir is equal to {} MWh.'.format(energy_stored_upper_res))
    time.sleep(small_sleeping_time)

    print('The wind speed on the island is equal to {} m/s.'.format(wind_speed_list_200[i]))
    time.sleep(small_sleeping_time)

    print('So the expected wind energy generation is equal to {} MWh.'.format(df_lesvos['Wind_Energy_Generated'][i]))
    time.sleep(small_sleeping_time)
    
    print('The Wind Energy Absorbed is equal to {} MWh.'.format(x*df_lesvos['Wind_Energy_Absorbed'][i]))
    time.sleep(small_sleeping_time)

    print('Hence, the wind energy rejected is equal to {} MWh.'.format(x*df_lesvos['Wind_Energy_Rejected'][i]))
    time.sleep(small_sleeping_time)
    
    
    fuel_cons_cost_in_kg = fuel_cons_cost * x*df_lesvos['Conv_Energy_To_Load'][i]
    
    tot_fuel_cons += fuel_cons_cost_in_kg

    print('At the present hour the generation of the conventional units is equal to {} MWh.'.format(x*df_lesvos['Conv_Energy_To_Load'][i]))
    time.sleep(small_sleeping_time)

    print('Since the fuel consumption is {} kg/MWh, the total consumption is equal to {} kg mazout.'.format(fuel_cons_cost, tot_fuel_cons ))
    
    time.sleep(small_sleeping_time)

    print('The total incomes of the investment are equal to {} €.'.format(initial_budget))

    time.sleep(small_sleeping_time)




    # STORING
    if df_lesvos['Wind_Energy_Rejected'][i] !=0:
        count_pump += pump_value

        print('Since the wind energy rejected is greater than zero, IT IS TIME TO STORE ENERGY!')
        time.sleep(small_sleeping_time)

        print('Hence, the system is storing the amount of {} MWh.'.format(x*df_lesvos['Wind_Energy_Rejected'][i]))
        time.sleep(small_sleeping_time)

        rejected_purchasing = cost_of_rej_wind * x*df_lesvos['Wind_Energy_Rejected'][i]

        print('The cost of purchasing rejected wind energy is equal to {} €/MWh, so the operators expenses are {} €.'.format(cost_of_rej_wind, rejected_purchasing))

        initial_budget -= rejected_purchasing

        energy_stored_upper_res += x*df_lesvos['Wind_Energy_Rejected'][i]

        time.sleep(small_sleeping_time)

        print('The energy stored in the upper reservoir is now equal to {} MWh.'.format(energy_stored_upper_res))

        
        time.sleep(small_sleeping_time)

        print('The total incomes of the investment are now equal to {} €.'.format(initial_budget))

        time.sleep(small_sleeping_time)


        # STORING / GENERATING

        if df_lesvos['Energy_Generated'][i] != 0:
            count_gen += gen_value

            print('The conventional generation along with the wind energy absorbed by the grid, can not meet the load demand,', end = ' ')
            time.sleep(small_sleeping_time)
            print('so the system should ALSO GENERATE ENERGY!')
            time.sleep(small_sleeping_time)

            print('STATE : GENERATE / STORE')
            time.sleep(small_sleeping_time)

            # WHEN THE ENERGY STORED IS ENOUGH TO PROVIDE GUARANTEED POWER

            if energy_stored_upper_res >= x*df_lesvos['Energy_Generated'][i]:

                print('The energy stored in the upper reservoir is sufficient for the guaranteed amount of energy generation, meaning there is no need to import energy from the grid!.')

                time.sleep(small_sleeping_time)

                print('Hence, the amount of energy being generated is equal to {} MWh.'.format(x*df_lesvos['Energy_Generated'][i]))
                time.sleep(small_sleeping_time)


                print('The operator is selling the guaranteed amount of energy at the price of {} €/MWh.'.format(cost_of_guar_energy))

                guar_revenues = cost_of_guar_energy * x * df_lesvos['Energy_Generated'][i]

                initial_budget += guar_revenues

                time.sleep(small_sleeping_time)

                print('The total incomes are now equal to {} €.'.format(initial_budget))

                energy_stored_upper_res -= x*df_lesvos['Energy_Generated'][i]

                time.sleep(small_sleeping_time)

                print('The energy stored in the upper reservoir is now equal to {} MWh.'.format(energy_stored_upper_res))


            # WHEN THE ENERGY STORED IS INSUFFICIENT TO PROVIDE GUARANTEED POWER

            else:
                print('The energy stored in the upper reservoir is INSUFFICIENT for the guaranteed amount of generation.')

                time.sleep(small_sleeping_time)

                print('Hence, the system should import the difference from the grid!')

                energy_imported_from_grid = x*df_lesvos['Energy_Generated'][i] - energy_stored_upper_res
                
                time.sleep(small_sleeping_time)

                print('The amount of energy being imported from the grid is equal to {} MWh.'.format(energy_imported_from_grid))

                grid_purchasing = cost_of_grid_import * energy_imported_from_grid

                energy_stored_upper_res += energy_imported_from_grid

                time.sleep(small_sleeping_time)

                print('The cost of purchasing energy from the grid is equal to {} €/MWh, so the operators expenses are {} €.'.format(cost_of_grid_import, grid_purchasing))               

                initial_budget -= grid_purchasing

                guar_revenues = cost_of_guar_energy * x * df_lesvos['Energy_Generated'][i]

                initial_budget += guar_revenues

                time.sleep(small_sleeping_time)



                print('The energy stored in the upper reservoir is now equal to {} MWh.'.format(energy_stored_upper_res))
                time.sleep(small_sleeping_time)
                
                print('The total incomes of the investment are now equal to {} €.'.format(initial_budget))


        # STORING / NOT GENERATING

        else:
            print('At the present time, the load demand is covered by the conventional generation and the wind energy absorbed.')
            time.sleep(small_sleeping_time)
            print('Hence, we should ONLY STORE ENERGY.')
            time.sleep(small_sleeping_time)

            print('STATE : STORE / NOT GENERATE')
            time.sleep(small_sleeping_time)

            count_gen = count_gen


    # NOT STORING

    else:
        count_pump = count_pump
        count_gen = count_gen

        print('The rejected wind energy is equal to zero,', end = ' ')
        time.sleep(small_sleeping_time)

        energy_stored_upper_res = energy_stored_upper_res

        print('meaning that the energy stored in the upper reservoir remains the same : {} MWh!'.format(energy_stored_upper_res))
        time.sleep(small_sleeping_time)

        

        # NOT STORING / GENERATING

        if df_lesvos['Energy_Generated'][i] != 0:
            count_gen += gen_value
            
            print('The conventional generation along with the wind energy absorbed by the grid, can not meet the load demand,', end = ' ')
            time.sleep(small_sleeping_time)

            print('hence, IT IS TIME TO GENERATE ENERGY!')
            time.sleep(small_sleeping_time)

            print('STATE : NOT STORE / GENERATE')
            time.sleep(small_sleeping_time)

            print('The energy stored in the upper reservoir is equal to {} MWh.'.format(energy_stored_upper_res))
            time.sleep(small_sleeping_time)

            # WHEN THE ENERGY STORED IN THE UPPER RESERVOIR IS ENOUGH TO PROVIDE GUARANTEED POWER!

            if energy_stored_upper_res >= x*df_lesvos['Energy_Generated'][i]:

                print('The energy stored in the upper reservoir is sufficient for the guaranteed amount generation, meaning there is no need to import energy from the grid!.')
                time.sleep(small_sleeping_time)

                print('Hence, the amount of energy being generated is equal to {} MWh.'.format(x*df_lesvos['Energy_Generated'][i]))
                time.sleep(small_sleeping_time)


                print('The operator is selling the guaranteed amount of energy at the price of {} €/MWh.'.format(cost_of_guar_energy))

                guar_revenues = cost_of_guar_energy * x * df_lesvos['Energy_Generated'][i]

                initial_budget += guar_revenues

                time.sleep(small_sleeping_time)

                print('The total incomes are now equal to {} €.'.format(initial_budget))

                energy_stored_upper_res -= x*df_lesvos['Energy_Generated'][i]

                time.sleep(small_sleeping_time)

                print('The energy stored in the upper reservoir is now equal to {} MWh.'.format(energy_stored_upper_res))
                time.sleep(small_sleeping_time)


            # WHEN THE ENERGY STORED IN THE UPPER RESERVOIR IS INSUFFICIENT TO PROVIDE GUARANTEED AMOUNT OF POWER!
            
            else:
                print('The energy stored in the upper reservoir is INSUFFICIENT for the guaranteed amount of generation.')

                time.sleep(small_sleeping_time)

                print('Hence, the system should import the difference from the grid!')

                energy_imported_from_grid = x*df_lesvos['Energy_Generated'][i] - energy_stored_upper_res
                
                time.sleep(small_sleeping_time)

                print('The amount of energy being imported from the grid is equal to {} MWh.'.format(energy_imported_from_grid))

                grid_purchasing = cost_of_grid_import * energy_imported_from_grid

                energy_stored_upper_res += energy_imported_from_grid

                time.sleep(small_sleeping_time)

                print('The cost of purchasing energy from the grid is equal to {} €/MWh, so the operators expenses are {} €.'.format(cost_of_grid_import, grid_purchasing))               

                initial_budget -= grid_purchasing

                guar_revenues = cost_of_guar_energy * x * df_lesvos['Energy_Generated'][i]

                initial_budget += guar_revenues 

                time.sleep(small_sleeping_time)

                print('Finaly, the amount of energy being generated is euqal to {} MWh.'.format(x*df_lesvos['Energy_Generated'][i]))

                time.sleep(small_sleeping_time)

                energy_stored_upper_res -= x*df_lesvos['Energy_Generated'][i]

                print('The energy stored in the upper reservoir is now equal to {} MWh.'.format(energy_stored_upper_res))
                time.sleep(small_sleeping_time)
                
                print('The total incomes of the investment are now equal to {} €.'.format(initial_budget))

        
        # NOT STORING / NOT GENERATING

        else:
            count_stand += stand_value
            count_pump = count_pump
            count_gen = count_gen


            print('At the present time, the load demand is covered by the conventional generation and the wind energy absorbed.')
            time.sleep(small_sleeping_time)
            
            print('Hence, the system does not need neither to generate or store energy.')
            time.sleep(small_sleeping_time)

            print('STATE : STAND-BY')
            initial_budget = initial_budget
            
            time.sleep(small_sleeping_time)

            print('The total incomes remain the same: {} €.'.format(initial_budget))
        
        time.sleep(sleeping_time)
    print('NEXT HOUR...')
    print('')




gen_pump_together = count_gen + count_pump + count_stand - 8760

print('Tot_hours_pumping : {} h'.format(round(count_pump)))
print('')
print('Tot_hours_generating : {} h'.format(round(count_gen)))
print('')
print('Tot_hours_standing : {} h'.format(round(count_stand)))
print('')
print('Tot_hours_gen&pump : {} h'.format(round(gen_pump_together)))
print('')
print('Total_Energy_Stored : {} MWh'.format(tot_stored))
print('')
print('Energy_Stored_Grid : {} MWh'.format(energy_stored_grid))
print('')
print('Energy_Stored_Wind : {} MWh'.format(energy_stored_wind))
print('')
print('Total_Energy_Generated : {} MWh'.format(tot_gen))
print('')
print('Objective Value : {}'.format(obj_value))
print('')
print('Optimal_installed_phes : {} MW'.format(gen_installed))
print('')
print('Optimal_installed_pump : {} MW'.format(pump_installed))
print('')
print('Optimal_upper_res_capacity : {} m3'.format(v_up_res))



    # if df_lesvos['Energy_Stored'][i] != 0 and df_lesvos['Energy_Generated'][i] == 0 :
        
    #     print('At the present time, the load demand is covered by the conventional generation and the wind energy absorbed.')
    #     print('Hence, it is time to store energy!')
    #     time.sleep(small_sleeping_time)

    #     if df_lesvos['Wind_Energy_Rejected'][i] != 0:
    #         print('Since the rejected wind energy is greater than zero,', end = ' ')
    #         print('the system is storing the rejected wind energy.')

    #         time.sleep(small_sleeping_time)

    #         print('So the amount of energy being stored, at the time being, is equal to {} MWh.'.format(x*df_lesvos['Energy_Stored'][i]))

            
    #         time.sleep(small_sleeping_time)

    #         rejected_purchasing = cost_of_rej_wind * x*df_lesvos['Energy_Stored'][i]

    #         print('The cost of purchasing rejected wind energy is equal to {} €/MWh, so the operators expenses are {} €.'.format(cost_of_rej_wind, rejected_purchasing))
    #         initial_budget -= rejected_purchasing
    #         time.sleep(small_sleeping_time)

    #         print('The total incomes of the investment are now equal to {} €.'.format(initial_budget))

            
    #     else:
    #         print('Since the rejected wind energy is equal to zero,', end = ' ')
    #         print('The system shall import energy from the grid.')
    #         time.sleep(small_sleeping_time)

    #         print('So the amount of energy being stored is equal to {} MWh.'.format(x*df_lesvos['Energy_Stored'][i]))
    #         time.sleep(small_sleeping_time)

    #         grid_purchasing = cost_of_grid_import * x * df_lesvos['Energy_Stored'][i]

    #         print('The cost of purchasing energy from the grid is equal to {} €/MWh, so the operators expenses are {} €.'.format(cost_of_grid_import, grid_purchasing))

    #         initial_budget -= grid_purchasing
    #         time.sleep(small_sleeping_time)
            
    #         print('The total incomes of the investment are now equal to {} €.'.format(initial_budget))

    # elif df_lesvos['Energy_Generated'][i] != 0 and df_lesvos['Energy_Stored'][i] == 0:

    #     print('The conventional generation along with the wind energy absorbed by the grid, can not meet the load demand,', end = ' ')
    #     time.sleep(small_sleeping_time)
    #     print('so it is time to generate energy!')
    #     time.sleep(small_sleeping_time)

    #     print('Hence, the amount of energy being generated is equal to {} MWh.'.format(x*df_lesvos['Energy_Generated'][i]))
    #     time.sleep(small_sleeping_time)

    #     print('The operator is selling the guaranteed amount of energy at the price of {} €/MWh.'.format(cost_of_guar_energy))

    #     guar_revenues = cost_of_guar_energy * x * df_lesvos['Energy_Generated'][i]

    #     initial_budget += guar_revenues

    #     time.sleep(small_sleeping_time)

    #     print('The total incomes are now equal to {} €.'.format(initial_budget))
        
    # elif df_lesvos['Energy_Generated'][i] == 0 and df_lesvos['Energy_Stored'][i] == 0:

    #     print('The system is standing-by')
    #     time.sleep(small_sleeping_time)

    #     print('So the amount of energy being stored and generated is equal to 0 MWh.')

    #     initial_budget = initial_budget
        
    #     time.sleep(small_sleeping_time)

    #     print('The total incomes remain the same, {} €.'.format(initial_budget))
    
    # time.sleep(sleeping_time)






    





