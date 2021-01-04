#%% Importing libraries
import pandas as pd
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
import datetime 
from pyomo.environ import *
from pyomo.opt import SolverFactory
#%% Importing data
# Loading csv
fields = ["Price", "Hour"]
prices_df = pd.read_csv('Prices_2019.csv', sep=';', usecols=fields, parse_dates=[1])
# Setting interval
init = '2019-01-01 00:00:00'  # First hour to appear
init_index = np.where(prices_df["Hour"] == init)[0][0]
end = '2019-12-31 23:00:00'  # Last hour to appear
end_index = np.where(prices_df["Hour"] == end)[0][0] + 1

# Generating list with prices and hours
prices = []
hours = []
for i in range(init_index, end_index):
    prices.append(prices_df.iloc[i, 0])
    hours.append(prices_df.iloc[i, 1])

# Creating datasets
train = 100
test = 1
prices_train = list(prices[0:24 * train])
prices_test = list(prices[24 * train:24 * (train + test)])

# Generating list with dates
date_init = datetime.datetime.fromtimestamp(1546297200)
Dates = []
for i in range(end_index, end_index+24*test):
    Dates.append(datetime.datetime.fromtimestamp(1546297200 + 3600*i).hour + 
				 datetime.datetime.fromtimestamp(1546297200 + 3600*i).minute)

#%% Price forecast function
def pricepred(p,d,q,P,D,Q, X_train):
    # Setting model order
    model_order = (p,d,q)
    model_seasonal_order = (P,D,Q,24)
    # Defining model
    model = sm.tsa.statespace.SARIMAX(X_train, order=model_order, seasonal_order=model_seasonal_order)
    model_fit = model.fit(disp=0)
    # Getting forecasted prices
    prediction = model_fit.forecast(steps=24)
     
    return prediction

#%% Arbitrage function
# BESS parameters
# -- Parameters --
Batt_Enom = 1					# [MWh] Battery nominal capacity
Batt_Pnom = 1.5					# [MW] Battery nominal power
Batt_ChEff = 0.9				# BESS charging efficiency
Batt_DchEff = 0.9			    # BESS discharging efficiency
Batt_Cost= 20000               	# [€] BESS cost
Batt_Eff = 0.9					# Provisional Battery efficiency
Batt_SOC_init = 0				# Initial SOC


# Function
def arbitrage(initial_SOC, energy_price, batt_capacity, batt_maxpower, 
              batt_efficiency, cost):
	# Model initialization
	model = ConcreteModel()
	model.time = range(24)
	model.time2 = range(1, 24)
	model.time3 = range(25)
	model.SOC = Var(model.time3, bounds=(0, batt_capacity), initialize=0)            # Battery SOC
	model.not_charging = Var(model.time, domain=Binary)                # Charge verifier
	model.not_discharging = Var(model.time, domain=Binary)             # Discharge verifier
	model.ESS_C = Var(model.time, bounds=(0, batt_maxpower))           # Energy being charged
	model.ESS_D = Var(model.time, bounds=(0, batt_maxpower))           # Energy being discharged
	model.DOD = Var(bounds=(0,100))
	model.cycles = Var(bounds=(500, 10000))
	model.max_SOC = Var(bounds=(initial_SOC, 100))
	model.min_SOC = Var(bounds=(0, initial_SOC))
	
	# Degradation model
	DOD_index = [0., 10., 10., 20., 20., 30., 30., 40., 40., 
			  50., 50., 60., 60.,70., 70.,80., 80., 90., 90., 100]
	cycles_index = [10000., 10000., 15000., 15000., 7000., 7000., 3300., 3300.,
				  2050., 2050., 1475., 1475., 1150., 1150., 950., 950., 
				  760., 760., 675., 675., 580., 580., 500., 500]
	
	
	model.deg=  Piecewise(model.cycles, model.DOD, # range and domain variables
                      pw_pts=DOD_index ,
                      pw_constr_type='EQ',
                      f_rule=cycles_index,
                      pw_repn='INC')
	
	# Defining the optimization constraints
	def c1_rule(model, t1):  # Checks there is enough room when charging
		return (batt_maxpower * model.not_charging[t1]) >= model.ESS_C[t1]
	model.c1 = Constraint(model.time, rule=c1_rule)
	
	def c2_rule(model, t1):  # Checks there is enough power when discharging
		return (batt_maxpower * model.not_discharging[t1]) >= model.ESS_D[t1]
	model.c2 = Constraint(model.time, rule=c2_rule)
	
	def c3_rule(model, t1):  # Prevents orders of charge and discharge simultaneously
		return (model.not_charging[t1] + model.not_discharging[t1]) <= 1
	model.c3 = Constraint(model.time, rule=c3_rule)
	
	def c4_rule(model, t2):  # The SOC must be the result of (SOC + charge*eff - discharge/eff) on the previous hour
		return model.SOC[t2] == (model.SOC[t2 - 1] + (model.ESS_C[t2 - 1] *
		                                              batt_efficiency - model.ESS_D[t2 - 1] / batt_efficiency))
	model.c4 = Constraint(model.time2, rule=c4_rule)
	
	def c5_rule(model):  # SOC at hour 0 must be the initial SOC
		return model.SOC[0] == initial_SOC
	model.c5 = Constraint(rule=c5_rule)
	   
	def c6_rule(model, t2):
		   return model.max_SOC >= model.SOC[t2] * (100 // Batt_Enom) 
	model.c6 = Constraint(model.time2, rule=c6_rule)

	def c7_rule(model, t2):
		return model.min_SOC <= model.SOC[t2] * (100 // Batt_Enom) 
	model.c7 = Constraint(model.time2, rule=c7_rule)
	
	def c8_rule(model):
		return model.DOD == model.max_SOC - model.min_SOC
	model.c8 = Constraint(rule=c8_rule)
	
	# Objective Function: Maximize profitability
	model.obj = Objective(
		expr=sum(((energy_price[t1] * (model.ESS_D[t1] - model.ESS_C[t1]))
			 - cost/model.cycles for t1 in model.time)), sense=maximize)
	    
	    
	
	# Applying the solver
	opt = SolverFactory('ipopt')
	opt.solve(model)
	# model.pprint()
	
	# Extracting data from model
	_E = [model.SOC[t1]() for t1 in model.time]
	_SOC = [i * (100 // Batt_Enom) for i in _E]
	_SOC.append(0)
	_P_output = np.zeros(len(energy_price))
	for i in range(len(energy_price)):
		if i == 0:
			_P_output[i] = 0
		else:
			_P_output[i-1] = round(_E[i] - _E[i-1],4)
    
	# Removes last discharge command if SOC at the end of the day is zero
# 	if _SOC[-1] == 0:
# 		_P_output[-1] = 0
	return _P_output

#%% Benefit function
def ben_calculator(powers, prices):
    benefits = []
    for i in range(len(prices)):
    	if i == 0:
    		Benh = 0
    	else:
    		Benh = -powers[i] * prices[i-1]
    	benefits.append(round(Benh,2))
    return sum(benefits)

#%% Simulation function
def simulator(price_pred, price_real):
    # Performing arbitrage with forecasted prices
    BESS_P = arbitrage(Batt_SOC_init, price_pred, Batt_Enom, Batt_Pnom, 
              Batt_ChEff, Batt_Cost)
    # Obtaining expected and real benefits
    Ben_exp = ben_calculator(BESS_P, price_pred)
    Ben_real = ben_calculator(BESS_P, prices_test)
    Ben_err = Ben_real - Ben_exp
    # Obtaining price forecast mean error
    price_for_err = [price_real[i]-price_pred[i] for i in range(len(prices_test))]
    Pred_err = sum(price_for_err) * 1.0/len(prices_test)
    return Ben_exp, Ben_real, Ben_err, Pred_err
#%% Scenario generator
# ENGINE SWITCH
scenario_generator = False
# ARIMA fixed parameters
d = 1
D = 1
P = 1
Q = 1
scenarios = {}
i = 1
if scenario_generator == True:   # If enabled generated 100 scenarios
    scengen_start = datetime.datetime.now()
    for p in range(1, 11):
        for q in range(1, 11):
            print("Generating scenario for SARIMA({},1,{}),(1,1,1,24)".format(p,q))
            ARIMA_start = datetime.datetime.now()
            prediction = pricepred(p,d,q,P,D,Q,prices_train)
            scenarios["{}".format(i)] = prediction
            ARIMA_end = datetime.datetime.now()
            ARIMA_duration = ARIMA_end - ARIMA_start
            print("Elapsed time: {}".format(ARIMA_duration))
            i = i+1
    np.save('scenarios.npy', scenarios)    
    scengen_end = datetime.datetime.now()
    scengen_duration = scengen_end - scengen_start
    print("Scenario generation time: {}".format(scengen_duration))

if scenario_generator == False: # When disabled loads scenarios.npy file
    scenarios = np.load('scenarios.npy',allow_pickle='TRUE').item()
    
#%% Simulation engine
# ENGINE SWITCH
simulation_engine = True
if simulation_engine:
    nscenarios = len(scenarios)
    Results = {'scenario':[], 'Ben_exp':[], 'Ben_real':[], 'Pred_err':[], 'Ben_err':[]}
    # Creating base case
    BESS_0 = arbitrage(Batt_SOC_init, prices_test, Batt_Enom, Batt_Pnom, 
                  Batt_ChEff, Batt_Cost)
    Ben_0 = ben_calculator(BESS_0, prices_test)
    Results['scenario'].append(0)
    Results['Ben_exp'].append(Ben_0)
    Results['Ben_real'].append(Ben_0)
    Results['Pred_err'].append(0)
    Results['Ben_err'].append(0)
    # Simulating scenarios
    for i in range(1, len(scenarios)+1):
        Ben_exp, Ben_real, Ben_err, Pred_err = simulator(scenarios["{}".format(i)], prices_test)
        Results['scenario'].append(i)
        Results['Ben_exp'].append(Ben_exp)
        Results['Ben_real'].append(Ben_real)
        Results['Pred_err'].append(Pred_err)
        Results['Ben_err'].append(Ben_err)
        
#%% Results analysis engine
figurecount = 0     # Figure window counter
# X axis dates label
dates_label = []
for i in range(24):
	dates_label.append('{}:00'.format(i))

# Plotting scenarios
for i in range(1, len(scenarios)+1):
    plt.figure(figurecount)
    plt.plot(scenarios["{}".format(i)], c=np.random.rand(3))
    plt.xlabel("Hour")
    plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
    plt.ylabel("Price (€/MWh)")
plt.plot(prices_test,linestyle='--', color='r', label='Real price')
plt.legend()
plt.grid()
plt.show()
figurecount = figurecount + 1
# Scenario evaluator
scenario_evaluator = True   # Switch for scenario evaluator
if scenario_evaluator:
    # Simulating scenario
    scenario = 50
    prices_scen = scenarios["{}".format(scenario)]
    Ben_exp, Ben_real, Ben_err, Pred_err = simulator(prices_scen, prices_test)
    Powers_scen =  arbitrage(Batt_SOC_init, prices_scen, Batt_Enom, Batt_Pnom, Batt_ChEff, Batt_Cost)
    # Plotting results
    plt.figure(figurecount)
    figurecount = figurecount + 1
    plt.plot(prices_scen, 'r', label='Forecast')
    plt.plot(prices_test, label='Real')
    plt.xlabel("Hour")
    plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
    plt.ylabel("Price (€/MWh)")
    plt.grid()
    plt.legend()
    plt.show()
    plt.figure(figurecount)
    figurecount = figurecount + 1
    plt.bar(len(prices_test), Powers_scen)
    plt.show()
    figurecount = figurecount + 1
# Results analysis
Price_errors = []
Ben_errors = []
for i in range(1, len(scenarios)+1): # Creating array with errors
    Price_errors.append(Results['Pred_err'][i])
    Ben_errors.append(Results['Ben_err'][i])

# Price errors distribution
fig = plt.figure(figurecount)
figurecount = figurecount + 1
n, bins, patches = plt.hist(Price_errors, bins=len(scenarios), edgecolor='black')
plt.ylabel("Mean price error (€/MWh)")
plt.show()
plt.grid()

# Benefit errors distribution
fig = plt.figure(figurecount)
figurecount = figurecount + 1
n, bins, patches = plt.hist(Ben_errors, bins=len(scenarios), edgecolor='black')
plt.ylabel("Expected benefit error (€)")
plt.show()
plt.grid()

# Benefit errors vs Price errors
fig = plt.figure(figurecount)
figurecount = figurecount + 1
plt.scatter(Ben_errors, Price_errors, s=2.5)
plt.xlabel('Expected benefit error (€)')
plt.ylabel('Price mean error (€/MWh)')
plt.grid()
plt.show()






