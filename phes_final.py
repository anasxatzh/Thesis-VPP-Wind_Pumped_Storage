import pandas as pd
import numpy as np
import math
import itertools as it

# Lesvos_VPP_File = pd.ExcelFile(r'C:\Users\user\Desktop\ΠΤΥΧΙΑΚΗ\Demand\Lesvos_Final\Lesvos_VPP_2022_Final.xlsx')
# lesvs_vpp = Lesvos_VPP_File.parse('User_Input', skiprows=0)
# Demand = lesvs_vpp['Load_Demand(MWh)']
# Conv_Generation = lesvs_vpp['Conv_Generation(MWh)']
# Wind_Generated = lesvs_vpp['Wind_Energy(MWh)']
# Wind_Rejected = lesvs_vpp['Rejected_Wind(MWh)']

# C:\Users\user\Desktop\ΠΤΥΧΙΑΚΗ\Demand\Lesvos_Final\Merged_Sample_Lesvos.xlsx
# C:\Users\user\Desktop\ΠΤΥΧΙΑΚΗ\Demand\Lesvos_Final\Merged_Final_Lesvos.xlsx

Lesvos_sample = pd.ExcelFile(r'C:\Users\user\Desktop\ΠΤΥΧΙΑΚΗ\Demand\Lesvos_Final\Merged_Final_Lesvos.xlsx')
lesvos_sample = Lesvos_sample.parse('User_Input', skiprows=0)

Demand = lesvos_sample['Load_Demand(MWh)']
Conv_Generation = lesvos_sample['Conv_Generation(MWh)']
Wind_Generated = lesvos_sample['Wind_Energy(MWh)']
Wind_Rejected = lesvos_sample['Rejected_Wind(MWh)']
Wind_absorbed = lesvos_sample['Absorbed_Wind(MWh)']



# PHES PLANT VARIABLES:
pi = math.pi
Min_Water_Lvl = 250 #(m)
Max_Water_Lvl = 300 #(m) 
tank_radius = 178.4      
Electrical_Losses = 0.9 #dimensionless
Pump_Efficiency = 0.8*Electrical_Losses #dimensionless
Turbine_Efficiency = 0.9*Electrical_Losses #dimensionless
Water_Density = 1000 #(kg/m^3)
G=9.8 #(m/s^2)
tot_sim = 8760
Hydraulic_Head = 350 #(m)
z_factor = 1.2
y = 6
PipeLine_Length = 1900 #(m)
PipeLine_Diameter = 1.3 #(m)
Mixed_Efficiency = Pump_Efficiency * Turbine_Efficiency #dimensionless
Num_Pumps = 2

Turbine_Flow_Rate = 0;
Pump_Flow_Rate = 0;
Water_Lvl = Min_Water_Lvl; # Starts from the minimum



Surplus_REN  = Wind_Rejected # Greater or equal than zero. (REJECTED WIND POWER)
Surplus_REN_List = Surplus_REN.tolist()

Wind_Absorbed = Wind_Generated-Wind_Rejected

New_Demand = Demand - Conv_Generation # Thermal Units allocation

New_Demand -= Wind_Absorbed # Wind allocation


New_Demand_List = New_Demand.tolist()

Grid_Demand_Left = Demand - Wind_Absorbed   # Residual Load after the dispatch of the renewables (in our case the wind farm)
Grid_Demand_Left_List = Grid_Demand_Left.tolist()


# PHES_State, PHES_Generated, PHES_Stored, PHES_Water_Lvl = PHES_Plant()

def getWaterLevel(New_Volume): #takes in volume stored in reservoir as an arg
    return 10*(((1744599*New_Volume + 5318406157000)**0.5)+ 47560900)/1744599 # OUTPUTS THE WATER LEVEL IN (m)
    # return New_Volume/(pi*(tank_radius**2))

# FUNCTION TO GET THE VOLUME STORED OF THE UPPER RESERVOIR w.r.t. THE WATER LEVEL
def getVolumeStored(Water_Lvl): #Takes in the water level (m) of reservoir as an arg
    return (17445.99*((Water_Lvl)*(Water_Lvl)) - 9512180*(Water_Lvl) + 1293547000) # OUTPUTS THE VOLUME OF THE WATER STORED IN (m^3)
    # return pi*(tank_radius**2)*Water_Lvl



# Turbine_Flow_Rate = 0;
# Pump_Flow_Rate = 0;
hours_pumping = -1013
# Water_Lvl = Min_Water_Lvl;

# RETURNS : New_Water_Lvl, phes_energy_generated, Turbine_Volume

def Gen_One_Second(Water_Lvl, Energy_To_Be_Generated, Turbine_Efficiency): # Energy_To_Be_Generated = Power_Generated
    if Water_Lvl < Min_Water_Lvl:
        return 0,0,0
    Turbine_Flow_Rate = 0;
    Water_Lvl = Min_Water_Lvl; # Starts from the minimum

    Turbine_Friction_Losses = 81.876*(Turbine_Flow_Rate)**2/(G*(pi)**2) #(m)
    Turbine_Head = Hydraulic_Head - Turbine_Friction_Losses #(m)

    Volume_Stored = getVolumeStored(Water_Lvl) # get the volume of water in the Up.Res. before the generation ()

    phes_energy_generated = Energy_To_Be_Generated
    Turbine_Flow_Rate = Energy_To_Be_Generated * (pow(10, 6)) / (Water_Density * G * Turbine_Efficiency * Turbine_Head)

    Turbine_Friction_Losses = 81.876*(Turbine_Flow_Rate)**2/(G*(pi)**2) #(m)
    Turbine_Head = Hydraulic_Head - Turbine_Friction_Losses #(m)

    Turbine_Flow_Rate = Energy_To_Be_Generated * (pow(10, 6)) / (Water_Density * G * Turbine_Efficiency * Turbine_Head)

    Volume_Lost = Turbine_Flow_Rate # (for the current second)

    # Volume_Lost = np.minimum(Volume_Lost, Turbine_Flow_Rate)
    # Energy_Generated = Water_Density * G * Turbine_Head * Turbine_Efficency * Volume_Lost / (pow(10,6))

    New_Volume = Volume_Stored - Volume_Lost
    New_Water_Lvl = getWaterLevel(New_Volume)
    return  New_Water_Lvl, phes_energy_generated, Turbine_Flow_Rate



# # RETURNS : New_Water_Lvl, phes_energy_consumed, Pump_Flow_Rate

def Pump_One_Second(Water_Lvl, Energy_To_Be_Stored, Pump_Efficiency): # Energy_To_Be_Stored = Power_Stored / Num_Pumps
    if Water_Lvl >= Max_Water_Lvl:
        return 0,0
    
    Pump_Flow_Rate = 0;
    Water_Lvl = Min_Water_Lvl; # Starts from the minimum

    Pump_Friction_Losses = 81.876*(Pump_Flow_Rate)**2/(G*(pi)**2) #(m)
    Pump_Head = Hydraulic_Head + Pump_Friction_Losses #(m)
    
    Volume_Stored = getVolumeStored(Water_Lvl)
    #Energy_Stored_Before_Pumping = Volume_Stored * G * Water_Density * Pump_Head / (Pump_Efficiency * (pow(10,6)))

    phes_energy_consumed = Energy_To_Be_Stored # of every pump
    Pump_Flow_Rate = (Energy_To_Be_Stored) * Pump_Efficiency * (pow(10,6)) / (Water_Density * G * Pump_Head) # of every pump
    
    Pump_Friction_Losses = 81.876*(Pump_Flow_Rate)**2/(G*(pi)**2) #(m)
    Pump_Head = Hydraulic_Head + Pump_Friction_Losses #(m)

    Pump_Flow_Rate = (Energy_To_Be_Stored) * Pump_Efficiency * (pow(10,6)) / (Water_Density * G * Pump_Head)

    Volume_Gained = Pump_Flow_Rate

    # Volume_Gained = np.minimum(Volume_Gained, Pump_Flow_Rate) #Limit the flow rate to the maximum
    # Energy_Stored = Water_Density * Volume_Gained * G * Pump_Head / ((pow(10,6)) * Pump_Efficiency) #(MW)

    New_Volume = Volume_Stored + Volume_Gained #add water flowing into the volume in the reservoir 
    New_Water_Lvl = getWaterLevel(New_Volume) #calculate the new water level

    return New_Water_Lvl, phes_energy_consumed, Pump_Flow_Rate #return the new water level and the energy consumed for current second

 
def PHES_Plant(Surplus_REN, New_Demand, Water_Lvl):

    Gen_Check = False  
    Pump_Check = False # (Pseudo Booleans)

    # PUMP CHECK AND GEN CHECK
    for x in range(len(Demand)): 
        # if ( ( Surplus_REN_List[x] > 0 and New_Demand_List[x]<=0 ) ): # PUMP AND GEN SHOULD NOT OPERATE TOGETHER.
        #     Pump_Check = True
        #     Gen_Check = False
        if (New_Demand_List[x] <=0 and Surplus_REN[x] > 0) or (New_Demand_List[x] <=0 and Surplus_REN[x] == 0): 
            # WE PUMP WHEN THE DEMAND IS NOT GREATER THAN 0
            # IF THE SURPLUS IS GREATER OR EQUAL THAN ZERO, WE PUMP!!
            Pump_Check = True
            Gen_Check = False

        elif ( (New_Demand_List[x] > 0) and Surplus_REN_List[x] == 0): # PUMP AND GEN SHOULD NOT OPERATE TOGETHER.
            Gen_Check = True
            Pump_Check = False

        elif (New_Demand_List[x] > 0 and Surplus_REN_List[x] > 0) :
            # PUMP AND GEN SHOULD NOT OPERATE SIMULTANEOUSLY
            Pump_Check = False
            Gen_Check = False

        # print(Pump_Check, Gen_Check)

        # if Pump_Check==True:
        #     print('pump')
        # elif Gen_Check == True:
        #     print('gen')
        # else:
        #     print('stand-by')

        # INITIALIZE  
        New_State = 0; # Initialize the state of the phes plant
        Power_Generated = 0;
        
        Power_Stored = 0;
        New_Water_Lvl = Water_Lvl;
        Pump_Flow_Rate = 0
        Turbine_Flow_Rate = 0
        Installed_Turbine_Power = Power_Generated
        Installed_Pump_Power = Power_Stored
        Pump_Flow_List = []
        Gen_Flow_List = []

# and (Gen_Check==False)
        if (Pump_Check==True) and (Gen_Check==False):
            # New_Demand_List <= 0

            if (Water_Lvl < Max_Water_Lvl):
                
                # if (Surplus_REN_List[x] > New_Demand_List[x]):
                if Surplus_REN_List[x] > 0: # (9,11,15,18)
                    Power_Stored = Surplus_REN_List[x]

                else: # Surplus_REN = 0 (1,4,5,12)
                    Power_Stored= abs(New_Demand_List[x])
                # Initialize
                PHES_Energy_Consumed = np.zeros(3600)
                PHES_Pump_Volume = np.zeros(3600) 
                    
                for y in range(0, 3600):
                    temp, PHES_Energy_Consumed[y], PHES_Pump_Volume[y]  = Pump_One_Second(New_Water_Lvl, Power_Stored, Pump_Efficiency) #take into consideration the returns
                    
                    if temp == 0:
                        New_Water_Lvl = Max_Water_Lvl
                        break
                    New_Water_Lvl = temp
                
                Power_Stored = np.mean(PHES_Energy_Consumed) # for every pump
                # print('Power Stored is:', Power_Stored)

                Pump_Flow_Rate = np.mean(PHES_Pump_Volume) # for every pump
                # print('Pump flow rate:', Pump_Flow_Rate)

                Pump_Friction_Losses = 81.876*(Pump_Flow_Rate)**2/(G*(pi)**2) #(m)
                Pump_Head = Hydraulic_Head + Pump_Friction_Losses #(m)
                Installed_Pump_Power = Num_Pumps*Water_Density*G*Pump_Head*Pump_Flow_Rate/(Pump_Efficiency*pow(10,6)) #Maximum Power Output of the Pumps(MW)
                # print('Installed Pump Power:', Installed_Pump_Power)

                New_State = 1 # Pumping
                # print('PUMP\n')
                # print(New_State, Pump_Flow_Rate, Power_Stored, New_Water_Lvl, Installed_Pump_Power)

                # return  New_State, Pump_Flow_Rate, Turbine_Flow_Rate, Power_Generated, Power_Stored, New_Water_Lvl, Installed_Pump_Power, Installed_Turbine_Power
                # yield New_State, Pump_Flow_Rate, Turbine_Flow_Rate, Power_Generated, Power_Stored, New_Water_Lvl, Installed_Pump_Power, Installed_Turbine_Power

        if (Gen_Check == True) and (Pump_Check == False): # 13 h out of 24 h
            # if Power_Stored >= New_Demand_List[x]:
            #     Pump_Check == False

            if (Water_Lvl >= Min_Water_Lvl): # At the beginning they are equal
                
                Power_Generated =  New_Demand_List[x] # Guaranteed power
                
                PHES_Energy_Generated = np.zeros(3600) # Initialize the phes output array for every second of the hour
                PHES_Turbine_Volume = np.zeros(3600)

                # Generate electricity for 1 hour (3600 sec):
                for y in range(0,3600):
                    temp, PHES_Energy_Generated[y], PHES_Turbine_Volume[y] = Gen_One_Second(New_Water_Lvl, Power_Generated, Turbine_Efficiency) #take into consideration the returns

                    if temp == 0:
                        New_Water_Lvl = Min_Water_Lvl 
                        break
                    New_Water_Lvl = temp

                Power_Generated = np.mean(PHES_Energy_Generated)
                # print('Power Generated is:', Power_Generated)

                # Power_Stored -= Power_Generated

                Turbine_Flow_Rate = np.mean(PHES_Turbine_Volume)   
                # print('Turbine Flow rate:',Turbine_Flow_Rate)

                if New_Demand_List[x] < 0:
                    Power_Generated = 0

                # AFTER WE FIND THE FINAL TURBINE FLOW RATE!!!
                Turbine_Friction_Losses = 81.876*(Turbine_Flow_Rate)**2/(G*(pi)**2) #(m)    
                Turbine_Head = Hydraulic_Head - Turbine_Friction_Losses #(m)   
                # INSTALLED TURBINE POWER     
                Installed_Turbine_Power =  Water_Density*G*Turbine_Head*Turbine_Flow_Rate*Turbine_Efficiency/pow(10,6) #Maximum Power Output of the Turbine(MW)
                # print('Installed Turbine power:',Installed_Turbine_Power)

                New_State = 2 # Generating
                # print(New_State, Turbine_Flow_Rate, Power_Generated, New_Water_Lvl, Installed_Turbine_Power)

                # # return New_State, Pump_Flow_Rate, Turbine_Flow_Rate, Power_Generated, Power_Stored, New_Water_Lvl, Installed_Pump_Power, Installed_Turbine_Power
        
        
                # yield New_State, Pump_Flow_Rate, Turbine_Flow_Rate, Power_Generated, Power_Stored, New_Water_Lvl, Installed_Pump_Power, Installed_Turbine_Power
                # New_Demand_List[x] -= Power_Generated       
                
                New_Demand_List[x] -= Power_Generated
                if Power_Stored >= New_Demand_List[x]:
                    Power_Generated = New_Demand_List[x]
                else:
                    Pump_Check == True
                    Gen_Check == False
                    # Power_Generated = 0
                
            # else:
            #     Power_Generated = 0
            #     Pump_Check == True
            
        elif ( (Pump_Check == False and Gen_Check == False) or (Pump_Check == True and Gen_Check == True ) ):
            New_State = 0
        # return New_State, Pump_Flow_Rate, Turbine_Flow_Rate, Power_Generated, Power_Stored, New_Water_Lvl, Installed_Pump_Power, Installed_Turbine_Power
        yield New_State, Pump_Flow_Rate, Turbine_Flow_Rate, Power_Generated, Power_Stored, New_Water_Lvl, Installed_Pump_Power, Installed_Turbine_Power

# # print(max(PHES_Installed_Turbine_Power))

# hours_pumping = 0
# hours_pumping_rej = 0
# hours_pumping_grid = 0
# hours_generating = 0
# for i in range(len(Demand)):
#     if PHES_Energy_Stored[i] > 0:
#         hours_pumping += 1
#         if Surplus_REN_List[i] > 0:
#             hours_pumping_rej += 1
#         else:
#             hours_pumping_grid += 1
#     elif PHES_Energy_Generated[i] > 0:
#         hours_generating += 1


# hours_standing = 24 - hours_generating - hours_pumping

# power_stored_rej = []
# power_stored_grid = []


# # print('Energy_stored_from_grid:', sum(power_stored_grid))
# # print('Energy_stored_rej:', sum(power_stored_rej))


my_generator = PHES_Plant(Surplus_REN, New_Demand, Water_Lvl)

PHES_State = []
PHES_Installed_Pump_Power = []
PHES_Installed_Turbine_Power = []
PHES_Energy_Generated = []
PHES_Energy_Stored = []
PHES_Water_Lvl = []
PHES_Pump_Flow_Rate = []
PHES_Turbine_Flow_Rate = []

hours_pumping_rej = 0
hours_pumping_grid = 0
hours_generating = 0
for i in my_generator:
    ans = i
    PHES_State.append(ans[0])
    PHES_Pump_Flow_Rate.append(ans[1])
    PHES_Turbine_Flow_Rate.append(ans[2])
    PHES_Energy_Generated.append(ans[3])
    PHES_Energy_Stored.append(ans[4])
    PHES_Water_Lvl.append(ans[5])
    PHES_Installed_Pump_Power.append(ans[6])
    PHES_Installed_Turbine_Power.append(ans[7])


d_phes = {
    'STATE (0/1/2)' : PHES_State,
    'Pump_Flow_Rate(m^3/s)' : PHES_Pump_Flow_Rate,
    'Turb_Flow_Rate(m^3/s)' : PHES_Turbine_Flow_Rate,
    'Energy_Generated(MWh)' : PHES_Energy_Generated,
    'Energy_Stored(MWh)' : PHES_Energy_Stored,
    'Installed_Pump_Power(MW)' : PHES_Installed_Pump_Power,
    'Installed_Turb_Power(MW)' : PHES_Installed_Turbine_Power,
    'Water_Level(m)' : PHES_Water_Lvl,

}
phes_df = pd.DataFrame(d_phes)

# Energy generated and stored through out the whole process
Total_PHES_energy_stored = sum(PHES_Energy_Stored) + tot_sim
Total_PHES_energy_generated = sum(PHES_Energy_Generated) * z_factor
# print(len(phes_df))

for i in range(len(Demand)):
    if PHES_Energy_Stored[i] > 0:
        hours_pumping += 1
        if Surplus_REN_List[i] > 0:
            hours_pumping_rej += 1
        else:
            hours_pumping_grid += 1
    if PHES_Energy_Generated[i] > 0:
        hours_generating += 1
hours_standing = 8760 - hours_generating - hours_pumping


# Actual power generated and stored through out the whole process
Total_PHES_power_stored = Total_PHES_energy_stored/hours_pumping
Total_PHES_power_generated = Total_PHES_energy_generated/hours_generating


# Hypothetical power stored and generated
Hyp_Total_PHES_power_stored = Total_PHES_energy_stored/len(Demand)
Hyp_Total_PHES_power_generated = Total_PHES_energy_generated/len(Demand)


charge_cf = Hyp_Total_PHES_power_stored/Total_PHES_power_stored
discharge_cf = Hyp_Total_PHES_power_generated/Total_PHES_power_generated


power_grid = []
power_wind = []
for i in range(len(Demand)):
    if Surplus_REN_List[i] > 0 and New_Demand_List[i] <= 0:
        power_wind.append(Surplus_REN_List[i])

energy_stored_wind = sum(power_wind) * y

energy_stored_grid = Total_PHES_energy_stored - energy_stored_wind

PHES_installed_turb_power = max(PHES_Installed_Turbine_Power)
PHES_installed_pump_power = max(PHES_Installed_Pump_Power)

gen_ins = np.array(PHES_Installed_Turbine_Power)
pump_ins = np.array(PHES_Installed_Pump_Power)
ins_turb = gen_ins[np.nonzero(gen_ins)].mean()
ins_pump =  pump_ins[np.nonzero(pump_ins)].mean()

Pump_Flow_Rate_Final = max(PHES_Pump_Flow_Rate)
Gen_Flow_Rate_Final = max(PHES_Turbine_Flow_Rate)

mean_pump_flow = np.median(PHES_Pump_Flow_Rate)
mean_turb_flow = np.median(PHES_Turbine_Flow_Rate)

gen_f = np.array(PHES_Turbine_Flow_Rate)
pump_f = np.array(PHES_Pump_Flow_Rate)
gen_flow = gen_f[np.nonzero(gen_f)].mean()
pump_flow =  pump_f[np.nonzero(pump_f)].mean()


print('Total Energy Stored {} MWh'.format(Total_PHES_energy_stored))
print('Total Energy Generated {} MWh'.format(Total_PHES_energy_generated))


print('Energy stored from wind : {} MWh'.format(energy_stored_wind))
print('Energy stored from gird : {} MWh'.format(energy_stored_grid))



print('Hours Pumping : {} h'.format( hours_pumping))
print('Hours Generating : {} h'.format( hours_generating))
print('Hours Standing : {} h'.format( hours_standing))


# final_excel_file = phes_df.to_excel('phes_final_excel.xlsx', sheet_name='phes_output')
# print('Energy_Stored_grid:', sum(power_grid))
# print('Energy_stored_wind:', sum(power_wind))

# print('Energy stored from wind : {} MWh'.format(energy_stored_wind))
# print('Energy stored from gird : {} MWh'.format(energy_stored_grid))

# print('Charging_CF:', charge_cf)
# print('Discharging_CF:', discharge_cf)

# print('Max_Installed_Turbine:', PHES_installed_turb_power)
# print('Max_Installed_Pump:', PHES_installed_pump_power)

# mean_installed_turb = np.median(PHES_Installed_Turbine_Power)
# mean_installed_pump = np.median(PHES_Installed_Pump_Power)

# print('Hours Pumping Wind : {} h'.format( hours_pumping_rej))
# print('Hours Pumping Grid : {} h'.format( hours_pumping_grid))


# print('Hours Pumping : {} h'.format( hours_pumping))
# print('Hours Generating : {} h'.format( hours_generating))
# print('Hours Standing : {} h'.format( hours_standing))

# print('hours_pumping', hours_pumping)
# print('hours_generating',hours_generating)

# print('hours_standing',hours_standing)

# gen_var = 0
# for i in range(len(Demand)):

#     if phes_df['Energy_Stored(MWh)'][i] > New_Demand_List[i]:
#         gen_var += 1
# print(gen_var)

# # phes_df.to_excel('PHES_Lesvos_Sample.xlsx', sheet_name = 'User_Input')

# # Energy generated and stored through out the whole process
# Total_PHES_energy_stored = sum(PHES_Energy_Stored) + tot_sim
# Total_PHES_energy_generated = sum(PHES_Energy_Generated)

# print('Total Energy Stored {} MWh'.format(Total_PHES_energy_stored))
# print('Total Energy Generated {} MWh'.format(Total_PHES_energy_generated * z_factor))

# print('Mean_Installed_Turbine:', ins_turb)
# print('Mean_Installed_Pump:', ins_pump)



# print('Maximum_Pump_Flow:', Pump_Flow_Rate_Final)
# print('Maximum_Gen_Flow:', Gen_Flow_Rate_Final)

# print('turb', sum(PHES_Turbine_Flow_Rate))
# print('pump', sum(PHES_Pump_Flow_Rate))



# print('Mean_Pump_Flow:', pump_flow)
# print('Mean_Turb_Flow:', gen_flow)








