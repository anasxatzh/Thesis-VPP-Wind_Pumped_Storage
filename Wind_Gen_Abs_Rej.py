import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np
from regex import E
import seaborn as sns
from windrose import WindroseAxes
import matplotlib.cm as cm
from dataclasses import dataclass
from multiprocessing.sharedctypes import Value
from optparse import Values
from pstats import Stats
import csv
place='Λέσβος, Ελλάδα'
file5='C:/Users/user/Downloads/POWER_Point_Hourly_20210101_20211231_039d1355N_025d5812E_LST.csv'
df=pd.read_csv(file5, skiprows=10, quoting=csv.QUOTE_NONE)
# print(df)
# print(df.shape)
df['Datetime']=pd.to_datetime(dict(year=df.YEAR,
                                   month=df.MO,
                                   day=df.DY,
                                   hour=df.HR))
df.set_index('Datetime', inplace=True)
df=df[['MO', 'WD50M', 'WS50M']]
df.columns=['Month', 'Direction', 'Speed']
# df

wind_speed_list = []
for i in range(len(df)):
    x = df['Speed'][i]
    wind_speed_list.append(x)



def wind_speed_func():
    
    return wind_speed_list




# MINIMUM, MAXIMUM, MEDIAN AND MEAN WIND SPEED!
# print ("Minimum wind speed: ", df.Speed.min())
# print ("Maximum wind speed: ", df.Speed.max())
# print ("Median wind speed: ", df.Speed.median())
# print ("Mean wind speed: ", df.Speed.mean())

#YEARLY WIND SPEED IN LESVOS!

df['Speed'].plot(figsize=(12,6))
plt.ylabel('Ταχύτητα ανέμου (m/s)')
plt.title(f'Ταχύτητα ανέμου για το έτος 2021 στην Λέσβο')
plt.savefig('Wind speed in Lesvos in 2021.jpeg', dpi=300)
plt.show()



#DAILY AVERAGE WIND SPEED IN LESVOS!

df.resample("D")["Speed"].mean().plot()
plt.title("Μέση ημερήσια ταχύτητα ανέμου")
plt.ylabel("Ταχύτητα ανέμου (m/s)")
plt.show()



# #MONTHLY AVERAGE WIND SPEED IN LESVOS!

df.resample("M")["Speed"].mean().plot()
plt.title("Μέση μηνιαία ταχύτητα ανέμου")
plt.ylabel("Ταχύτητα ανέμου (m/s)")
plt.show()



#MONTHLY WIND SPEED IN 2021_LESVOS!

fig, ax = plt.subplots(2,6, sharey = True, figsize = (20, 8))
ax[0,0].plot(df[df.Month == 1]["Speed"])
ax[0,0].set_title("Ιανουάριος")
ax[0,0].set_xticks([])

ax[0,1].plot(df[df.Month == 2]["Speed"])
ax[0,1].set_title("Φεβρουάριος")
ax[0,1].set_xticks([])

ax[0,2].plot(df[df.Month == 3]["Speed"])
ax[0,2].set_title("Μάρτιος")
ax[0,2].set_xticks([])

ax[0,3].plot(df[df.Month == 4]["Speed"])
ax[0,3].set_title("Απρίλιος")
ax[0,3].set_xticks([])

ax[0,4].plot(df[df.Month == 5]["Speed"])
ax[0,4].set_title("Μάιος")
ax[0,4].set_xticks([])

ax[0,5].plot(df[df.Month == 6]["Speed"])
ax[0,5].set_title("Ιούνιος")
ax[0,5].set_xticks([])

ax[1,0].plot(df[df.Month == 7]["Speed"])
ax[1,0].set_title("Ιούλιος")
ax[1,0].set_xticks([])

ax[1,1].plot(df[df.Month == 8]["Speed"])
ax[1,1].set_title("Αύγουστος")
ax[1,1].set_xticks([])

ax[1,2].plot(df[df.Month == 9]["Speed"])
ax[1,2].set_title("Σεπτέμβριος")
ax[1,2].set_xticks([])
num_wind_turb = 17.48
ax[1,3].plot(df[df.Month == 10]["Speed"])
ax[1,3].set_title("Οκτώβριος")
ax[1,3].set_xticks([])

ax[1,4].plot(df[df.Month == 11]["Speed"])
ax[1,4].set_title("Νοέμβριος")
ax[1,4].set_xticks([])

ax[1,5].plot(df[df.Month == 12]["Speed"])
ax[1,5].set_title("Δεκέμβριος")
ax[1,5].set_xticks([])

ax[0,0].set_ylabel("Ταχύτητα ανέμου (m/s)")
ax[1,0].set_ylabel("Ταχύτητα ανέμου (m/s)")

plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

fig.suptitle(f"Μηνιαία ταχύτητα του ανέμου για την Λέσβο")
fig.supxlabel("Μήνες")

plt.savefig("Monthly Wind Speed_Lesvos_2021.jpeg",
           dpi = 300)

plt.show()



#WEIBULL DISTRIBUTION LESVOS!
from scipy import stats
data=df.Speed.values
data
params=stats.weibull_min.fit(data,
                             floc=0, #location=0
                             scale=2)
params

#ΠΑΡΑΓΟΝΤΑΣ ΜΟΡΦΗΣ (k)
shape_factor=params[0]
shape_factor
#2.342699144640077

#ΣΥΝΤΕΛΕΣΤΗΣ (c)
scale_parameter=params[2]
scale_parameter
#8.125415486809722


#SYNARTISI PYKNOTHTAS PITHANOTHTAS!
speed_range=np.arange(0, 29)
plt.plot(stats.weibull_min.pdf(speed_range, *params),
         color='blue',
         label='Συνάρτηση Πυκνότητας Πιθανότητας (PDF)')
plt.legend()
plt.show()


#ATHRISTIKI SINARTISI PITHANOTHTAS!
plt.plot(stats.weibull_min.cdf(speed_range, *params),
         color='red',
         label='Αθροιστικη Συνάρτηση Πιθανότητας (CDF)')
plt.legend()
plt.show()

#ΑΘΡΟΙΣΤΙΚΗ_ΣΥΝΑΡΤΗΣΗ & ΠΥΚΝΟΤΗΤΑ_ΠΙΘΑΝΟΤΗΤΑΣ!
plt.plot(stats.weibull_min.pdf(speed_range, *params),
         color='blue',
         label='Πυκνότητα Πιθανότητας')
plt.plot(stats.weibull_min.cdf(speed_range, *params),
         color='red',
         label='Αθροιστική Συνάρτηση')
plt.legend()
plt.show()


#SYNARTISI PYKNOTHTAS PITHANOTHTAS_PARAGONTAS_MORFHS(k)!
speed_range=np.arange(0,29)

plt.plot(stats.weibull_min.pdf(speed_range, *params),
         color = "blue",
        label = "k = 2.2")

plt.plot(stats.weibull_min.pdf(speed_range, *(1.5, 0, 8.15)),
         color = "green",
        label = "k = 1.5")

plt.plot(stats.weibull_min.pdf(speed_range, *(3, 0, 8.15)),
        label = "k = 3",
        color = "red")
plt.xlabel('Ταχύτητα (m/s)')
plt.ylabel('Συνάρτηση Πυκνότητας Πιθανότητας')
plt.title('Μεταβολή της Συνάρτησης Πυκνότητας Σύμφωνα με τον Παράγοντα Μορφής (k)')
plt.legend()
plt.show()

#WEIBULL!!
values, bins,hist = plt.hist(data, bins=28, rwidth=0.85, color='cornflowerblue')

speed_range = np.arange(0,29).tolist()

plt.plot(stats.weibull_min.pdf(speed_range, *params)*len(df),
         color='darkorange',
         label='Συνάρτηση Πυκνότητας Πιθανότητας')

plt.legend()
plt.show()


#SPEED_CLASS!
df['Speed Class']=''
for index in df.index:
    df.loc[index, 'Speed Class']=math.ceil(df.loc[index, 'Speed'])
# df


#WEIBULL_2!!
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["font.size"] = 14

sns.histplot(data = df,  
            x = df.Speed,
            kde = True, 
            bins = 29)

plt.ylabel("Αριθμός Ωρών")
plt.title(f"Κατανομή Weibull στην Λέσβο")

plt.savefig("Κατανομή Weibull_Lesvos",
           dpi = 300)
plt.show()

#SYXNOTHTA_EMFANISIS
df_new = df.groupby(['Speed Class']).count()
df_new = df_new[["Speed"]]
df_new.columns = ['Frequency']

df_new['Cumulative Frequency'] = df_new['Frequency'].cumsum()

df_new.sort_index(inplace = True)

df_new

# #ROIKO DIAGRAMMA!
from windrose import WindroseAxes
plt.figure(figsize = (15, 17))
ax = WindroseAxes.from_ax()

ax.bar(df.Direction,
       df.Speed,
    #    normed=True,    #get % of number of hours
       opening= 0.8,    #width of bars
       edgecolor='white',
      )

ax.set_legend(loc = "best")

plt.title(f"Ροϊκό Διάγραμμα για το νησί της Λέσβου")
plt.savefig("Lesvos wind rose diagram.jpeg",
           dpi = 300)

plt.show()



#KAMPYLH ISXYOS!
wind_speed = np.arange(1, 29).tolist()
wind_power = [0.0, 0.0, 1.7, 14.7, 40.8, 70.6, 134.8, 207.1, 292.6, 403.4, 508.1, 504.6, 613.2, 613.2, 613.2, 613.2, 613.2, 613.2, 613.2, 613.2, 613.2, 613.2, 613.2, 613.2, 613.2, 613.2, 613.2, 613.2]

#ENERCON E40-600kW
power_curve = pd.DataFrame({"Speed Class": wind_speed,
                           "Power at given speed":wind_power})

power_curve.set_index("Speed Class", inplace = True)
cut_in_speed = 2.5 
cut_out_speed = 28
rated_speed = 12
rated_power = 600
power_curve
num_turb = 15
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
power_curve["Hours"] = df_new.index.to_series().map(df_new["Frequency"])
power_curve["Frequency (%)"] = (power_curve["Hours"]/power_curve["Hours"].sum()) * 100
power_curve["Power production distribution"] = power_curve["Frequency (%)"] * power_curve["Power at given speed"]/(100 * 100)
power_curve["Energy yield"] = power_curve["Power at given speed"] * power_curve["Hours"]

power_curve = power_curve.dropna()

##power_curve['Speed Class'] = power_curve['Speed Class'].shift(1)
##power_curve = power_curve.fillna(1)

speed_class = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
power_curve['Speed Class'] = speed_class


# # WIND ENERGY DATAFRAME:

wind_list = np.array([0]*8760, dtype='float64')
for i in range(len(wind_list)):
    # i : 0-->8759
    for j in range(1,22):
        # j : 1-->21
        if df['Speed Class'][i] == power_curve['Speed Class'][j]:
            wind_list[i] = (power_curve['Energy yield'][j]/power_curve['Hours'][j])*pow(10,-3)*15

wind_list = wind_list.tolist()

# wind_df = pd.DataFrame(wind_list, columns=['Wind_Energy(MWh)'])
# wind_df['Wind_Energy(MWh)'] = wind_df['Wind_Energy(MWh)'].astype(float)
# # wind_df.to_excel('Hourly_Wind_Generation_Lesvos_2022.xlsx',sheet_name='User_Input')   


# # # DEMAND DATAFRAME:

Dem = pd.ExcelFile(r'C:\Users\user\Documents\Lesvos_Demand_final2.xlsx')
lesvos_demand = Dem.parse('User_Input', skiprows=0)
lesvos_dem = lesvos_demand['cons'].tolist()

demand_list = np.array([0]*8760,dtype='float64')
# demand_list = [] # len = 8760

i=0
k= 24
j=0
while ( k <= len(demand_list)  ) and ( i<=len(demand_list) - 24 ) and ( j <= len(lesvos_dem) ): 
    # k: 24-->48-->72-->...-->8760
    # i: 0-->24-->48-->72-->...-->8736
    # j: 0-->1-->2-->...-->365
    demand_list[i:k] = lesvos_dem[j]/24
    i += 24
    k += 24
    j += 1

demand_list = demand_list.tolist()
# df2 = pd.DataFrame(demand_list, columns=['Load_Demand'])
# df2['Load_Demand'] = df2['Load_Demand'].astype(float)


# # # CONVENTIONAL GENERATION DATAFRAME:
wind_energy_generated = power_curve["Energy yield"].sum() * num_wind_turb / 1000 # MW

Lesvos_Conv_dispatch_order = pd.ExcelFile(r'C:\Users\user\Desktop\ΠΤΥΧΙΑΚΗ\Demand\Lesvos_Conv_Dispatch_Order2.xlsx')
lesvos_dispatch = Lesvos_Conv_dispatch_order.parse('User_Input')
conv_production = lesvos_dispatch['Conv_Production(MWh)'].tolist()
conv_list = np.array([0]*8760,dtype='float64')


for i in range(4015,5840,24): # 15/06-->01/09
    # 4015-->4039-->4063-->...-->5839
    conv_list[i:i+2] = conv_production[1]
    conv_list[i+2:i+6] = conv_production[3]
    conv_list[i+6:i+8] = conv_production[1]
    conv_list[i+8:i+17] = conv_production[6]
    conv_list[i+17:i+24] = conv_production[8]
# print(conv_list[4015:4039])
# print(conv_production)

for i in range(5840,8030,24): # 01/09-->01/12
    # 4015-->4039-->4063-->...-->5839
    conv_list[i:i+2] = conv_production[0]
    conv_list[i+2:i+5] = conv_production[4]
    conv_list[i+5:i+17] = conv_production[0]
    conv_list[i+17:i+24] = conv_production[5]


for i in range(2190,4015,24): # 01/04-->15/06
    # 4015-->4039-->4063-->...-->5839
    conv_list[i:i+2] = conv_production[0]
    conv_list[i+2:i+5] = conv_production[4]
    conv_list[i+5:i+17] = conv_production[0]
    conv_list[i+17:i+24] = conv_production[5]

for i in range(8030,8760,24): # 01/12-->01/01
    # 4015-->4039-->4063-->...-->5839
    conv_list[i:i+5] = conv_production[1]
    conv_list[i+5:i+8] = conv_production[5]
    conv_list[i+8:i+24] = conv_production[6]
for i in range(0,24,24): # 01/1-->02/01
    # 4015-->4039-->4063-->...-->5839
    conv_list[i:i+5] = conv_production[1]
    conv_list[i+5:i+8] = conv_production[5]
    conv_list[i+8:i+24] = conv_production[6]

for i in range(24,1095,24): # 02/01-->15/02
    # 4015-->4039-->4063-->...-->5839
    conv_list[i:i+5] = conv_production[2]
    conv_list[i+5:i+16] = conv_production[7]
    conv_list[i+16:i+24] = conv_production[9]

for i in range(1095,2190,24): # 01/04-->15/06
    # 4015-->4039-->4063-->...-->5839
    conv_list[i:i+5] = conv_production[1]
    conv_list[i+5:i+12] = conv_production[5]
    conv_list[i+12:i+16] = conv_production[1]
    conv_list[i+16:i+24] = conv_production[7]

conv_list = conv_list.tolist()


# df = pd.DataFrame(conv_list, columns=['Conventional_Power'])
# df['Conventional_Power'] = df['Conventional_Power'].astype(float)

# # df.to_excel('Conventional_Generation_Lesvos_2022_ffinal.xlsx',sheet_name='User_Input')




# # FINAL DATAFRAME

# d_final = {'Wind_Energy(MWh)':wind_list,
#            'Load_Demand(MWh)':demand_list,
#            'Conv_Generation(MWh)':conv_list
#           }

# df2_final = pd.DataFrame(d_final)

# df_final.to_excel('Lesvos_2022_Final.xlsx',sheet_name='User_Input')    






# #ATHROISMA WRWN
power_curve['Hours'].sum()
#ATHROISMA ENERGEIAS
power_curve['Energy yield'].sum() # kW
rated_power = 613.2 # kW
wind_energy_generated = power_curve["Energy yield"].sum() * num_wind_turb / 1000 # MW
capacity_factor = wind_energy_generated / (power_curve["Hours"].sum() * rated_power * num_turb / 1000)
capacity_factor = round(capacity_factor, 4)


# # # REJECTED WIND ENERGY
tech_minima = 0.5
upper_bound = 0.2
frames_looping = 4
Lesvos_Final_File = pd.ExcelFile(r'C:\Users\user\Desktop\ΠΤΥΧΙΑΚΗ\Demand\Lesvos_Final\Lesvos_2022_Final.xlsx')
lesvos_final = Lesvos_Final_File.parse('User_Input',skiprows = 0)

wind_energy = lesvos_final['Wind_Energy(MWh)']
load_demand = lesvos_final['Load_Demand(MWh)']
conv_generation = lesvos_final['Conv_Generation(MWh)']

# Wind Energy Absorbed:
for i in range(len(load_demand)):
    if load_demand[i] <= conv_generation[i]*tech_minima:
        max_wind_abs = 0

    elif load_demand[i] >= conv_generation[i]*tech_minima and load_demand[i] <= (1+upper_bound)*conv_generation[i]*tech_minima:
        max_wind_abs = load_demand - conv_generation*tech_minima

    elif load_demand[i] >= (1+upper_bound)*conv_generation[i]*tech_minima:
        max_wind_abs = upper_bound*load_demand

# Rejected wind energy:
wind_energy_list = wind_energy.tolist()
rejected_wind = wind_energy - max_wind_abs
rejected_wind_list = rejected_wind.tolist()
rej_wind_list = []
for i in range(len(rejected_wind_list)):
    if rejected_wind_list[i] < 0:
        rejected_wind_list[i] = 0
    else:
        pass
absorbed_wind_list = []
for i in range(len(rejected_wind_list)):
    abs_wind = wind_energy_list[i] - rejected_wind_list[i]
    absorbed_wind_list.append(abs_wind)
rej_data = {
    'Rejected_Wind' : rejected_wind_list
}
rej_df = pd.DataFrame(rej_data)
frames_rej = []
for i in range(frames_looping):
    frames_rej.append(rej_df)
rej_wind_df = pd.concat(frames_rej)

wind_energy_rejected = rej_wind_df['Rejected_Wind'].sum()
wind_energy_absorbed = wind_energy_generated - wind_energy_rejected

d_final = {'Wind_Energy(MWh)':wind_list,
           'Load_Demand(MWh)':demand_list,
           'Conv_Generation(MWh)':conv_list,
           'Rejected_Wind(MWh)' : rejected_wind_list,
           'Absorbed_Wind(MWh)' : absorbed_wind_list,

          }
df_final = pd.DataFrame(d_final)


wind_data = {
           'Wind_Energy(MWh)':wind_list,
           'Rejected_Wind(MWh)' : rejected_wind_list,
           'Absorbed_Wind(MWh)' : absorbed_wind_list,

            }

wind_dataframe = pd.DataFrame(wind_data)
count_rej = 0
for i in range(len(df_final)):
    if df_final['Rejected_Wind(MWh)'][i] > 0:
        count_rej += 1


# df_final['Load_Demand(MWh)'].plot(figsize=(12,6))
# x = np.arange(0,8760)
# y = df2_final['Load_Demand(MWh)']
# plt.plot(x,y)
# plt.xticks(np.arange(0,8761,730))
# plt.ylabel('Ηλεκτρική Ζήτηση (MW)')
# plt.xlabel('Ώρες του χρόνου (h)')
# plt.title('Ζήτηση ηλεκτρικής ενέργειας')
# plt.ylim(10,60)
# plt.fill_between(x, 0, y, facecolor='grey')
# plt.show()

# df_final.to_excel('Lesvos_VPP_Final_Thesis_2022.xlsx',sheet_name='User_Input')  
df_final['date'] = pd.date_range(start = '01/01/2021', periods = len(df_final), freq='1H')
df_final = df_final.set_index('date')


# print (f"Ο συντελεστής χρησιμοποίησης στην τοποθεσία {place} το 2021 είναι ", capacity_factor*100, "%.")
# print('Total Wind Energy Generated (MWh) : {} MWh'.format(wind_energy_generated))
# print('Total Wind Energy Rejected (MWh) : {} MWh'.format(wind_energy_rejected))
# print('Total Wind Energy Absorbed (MWh) : {} MWh'.format(wind_energy_absorbed))
# print('Total hours rejecting : {} h'.format(count_rej))
# print(power_curve)

# print(df_final.head())
# print(len(df_final))



# PLOT REJECTED WIND!!
y = df_final['Rejected_Wind(MWh)']
x = np.arange(0,8760)
plt.plot(x,y, color='r')
plt.xticks(np.arange(0,8761,730))
plt.ylabel('Ισχύς (MW)')
plt.xlabel('Ώρες του χρόνου (h)')
plt.title('Απορριπτόμενη αιολική ισχύς')
plt.ylim(0,6)
plt.figure(figsize=(12,6))
plt.fill_between(x,0,y, facecolor='red')
# plt.show()



# PLOT REJECTED AND ABSORBED WIND!!
fig = plt.figure(figsize=(12,6))
x = np.arange(0,8760)
plt.plot(x, df_final['Rejected_Wind(MWh)'], label = 'Απορριπτόμενη', color = 'red')
plt.plot(x, df_final['Absorbed_Wind(MWh)'], label = 'Απορροφούμενη', color = 'blue')
plt.xticks(np.arange(0,8761,730))
plt.ylabel('Ισχύς (MW)')
plt.xlabel('Ώρες του χρόνου (h)')
plt.title('Απορροφούμενη & Απορριπτόμενη')
plt.ylim(0,15)
plt.tight_layout()
plt.legend()
# plt.show()

print(power_curve)

# y_dem = df_final['Load_Demand(MWh)']
# y_conv = df_final['Conv_Generation(MWh)']
# x_dem = np.arange(0,8760)
# x_conv = np.arange(0,8760)
# plt.plot(x_dem, y_dem, label = 'Ηλεκτρική Ζήτηση')
# plt.plot(x_conv, y_conv, label = 'Θερμική Παραγωγή')
# plt.xlabel('Ώρες του χρόνου (h)')
# plt.ylabel('Ισχυς (MW)')
# plt.xticks(np.arange(0,8761,730))
# plt.legend()
# plt.fill_between(x_dem, 0, y_dem, facecolor='black')
# plt.fill_between(x_conv, 0, y_conv, facecolor='grey')
# plt.tight_layout()
# plt.figure(figsize=(12,6))
# plt.show()


# X = list(df_final.iloc[:,4])
# Y = list(df_final.iloc[:,3])

# sns.lineplot(x=X,y=Y)
# plt.tick_params(axis = 'x', labelsize = 15, rotation = 90)



# # df_sample = df_final.sample(n=24)
# # df_sample.to_excel('Lesvos_VPP_Sample.xlsx', sheet_name='User_Input')
# # print(df_sample)