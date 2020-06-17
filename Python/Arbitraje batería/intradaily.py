# --- Standalone battery doing 48-hour arbitrage schedule and changing it schedule on every session
# of day D-1's intradaily market---
#
# Author: Pedro Luis Camuñas
# Date: 29/04/2020

from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand

# -- Simulation parameters --
Batt_Emax = 4                       # Rated battery energy (MWh)
Batt_Pmax = 2             # Rated battery power (MW)
Batt_Efficiency = 0.9               # Rated battery efficiency

# -- Data initialization --
SOC_i = 0                           # Initial battery SOC

# Initializing the daily market schedule with zeros
# Schedule_DM = []
# for i in range(24):
# 	Schedule_DM.append(0)

# -- Data importing --
data = pd.read_excel('Prices.xlsx', sheet_name='Prices', nrows=200)  # TODO: Change to csv
Price_real = list(data['Price'][0:48])
Price_0 = Price_real[0:24] + [i + i*rand.uniform(-0.14, 0.14) for i in Price_real[24:48]]
Price_1 = Price_real[0:24] + [i + i*rand.uniform(-0.12, 0.12) for i in Price_real[24:48]]
Price_2 = Price_real[0:24] + [i + i*rand.uniform(-0.1, 0.1) for i in Price_real[24:48]]
Price_3 = Price_real[0:24] + [i + i*rand.uniform(-0.08, 0.08) for i in Price_real[24:48]]
Price_4 = Price_real[0:24] + [i + i*rand.uniform(-0.06, 0.06) for i in Price_real[24:48]]
Price_5 = Price_real[0:24] + [i + i*rand.uniform(-0.04, 0.04) for i in Price_real[24:48]]
Price_6 = Price_real[0:24] + [i + i*rand.uniform(-0.02, 0.02) for i in Price_real[24:48]]
Price_7 = Price_real

# -- Battery scheduler --
def scheduler(initial_SOC, energy_price, batt_capacity, batt_maxpower, batt_efficiency, schedule_range):
	# Model initialization
	model = ConcreteModel()
	model.time = range(schedule_range)
	model.time2 = range(1, schedule_range)
	model.time3 = range(schedule_range+1)
	model.SOC = Var(model.time3, bounds=(0, batt_capacity))         # Battery SOC
	model.not_charging = Var(model.time, domain=Binary)             # Charge verifier
	model.not_discharging = Var(model.time, domain=Binary)          # Discharge verifier
	model.ESS_C = Var(model.time, bounds=(0, batt_maxpower))        # Energy being charged
	model.ESS_D = Var(model.time, bounds=(0, batt_maxpower))        # Energy being discharged

	# Defining the optimization constraints
	def c1_rule(model, t1):  # Checks there is enough energy when charging
		return (batt_capacity * model.not_charging[t1]) >= model.ESS_C[t1]
	model.c1 = Constraint(model.time, rule=c1_rule)

	def c2_rule(model, t1):  # Checks there is enough energy when discharging
		return (batt_capacity * model.not_discharging[t1]) >= model.ESS_D[t1]
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

	# Objective Function: Maximize profitability
	model.obj = Objective(
		expr=sum(((energy_price[t1] * (model.ESS_D[t1] - model.ESS_C[t1]))
		          for t1 in model.time)), sense=maximize)

	# Applying the solver
	opt = SolverFactory('cbc')
	opt.solve(model)
	model.pprint()

	# Extracting data from model
	P_output = [-model.ESS_D[t1]() + model.ESS_C[t1]() for t1 in model.time]
	SOC = [model.SOC[t1]() for t1 in model.time]
	if SOC[-1] == 0:
		P_output[-1] = 0

	return P_output, SOC
#
# # Session 0
Powers_0, SOC_0 = scheduler(SOC_i, Price_0, Batt_Emax, Batt_Pmax, Batt_Efficiency, len(Price_0))
CashFlow_0_real = sum([(-a * b) for a, b in zip(Powers_0, Price_real)])
print(len(Price_0))
print('- Session 0, daily market D (scheduled D + D1 with 14% error): {} €'.format(round(CashFlow_0_real, 2)))

# Session 1
Powers_1, SOC_1 = scheduler(SOC_i, Price_1, Batt_Emax, Batt_Pmax, Batt_Efficiency, len(Price_1))
CashFlow_1_real = sum([(-a * b) for a, b in zip(Powers_1, Price_real)])
print('- Session 1 (re-scheduled D + D1 with  12% error): {} €'.format(round(CashFlow_1_real, 2)))

# Session 2
Powers_2, SOC_2 = scheduler(SOC_i, Price_2, Batt_Emax, Batt_Pmax, Batt_Efficiency, len(Price_2))
CashFlow_2_real = sum([(-a * b) for a, b in zip(Powers_2, Price_real)])
print('- Session 2 (re-scheduled D + D1 with 10% error): {} €'.format(round(CashFlow_2_real, 2)))

# Session 3
Powers_3, SOC_3 = scheduler(SOC_i, Price_3, Batt_Emax, Batt_Pmax, Batt_Efficiency, len(Price_3))
CashFlow_3_real = sum([(-a * b) for a, b in zip(Powers_3, Price_real)])
print('- Session 3 (re-scheduled D + D1 with 8% error): {} €'.format(round(CashFlow_3_real, 2)))

# Session 4
Powers_4, SOC_4 = scheduler(SOC_3[5], Price_4[5:len(Price_4)], Batt_Emax, Batt_Pmax, Batt_Efficiency, 43)
Powers_4 = Powers_3[0:5] + Powers_4
SOC_4 = SOC_3[0:5] + SOC_4
CashFlow_4_real = sum([(-a * b) for a, b in zip(Powers_4, Price_real)])
print('- Session 4 (re-scheduled 5-24h of D  & D1 complete D+1 with 6% error): {} €'.format(round(CashFlow_4_real, 2)))

# Session 5
Powers_5, SOC_5 = scheduler(SOC_3[8], Price_5[8:len(Price_5)], Batt_Emax, Batt_Pmax, Batt_Efficiency, 40)
Powers_5 = Powers_4[0:8] + Powers_5
SOC_5 = SOC_4[0:8] + SOC_5
CashFlow_5_real = sum([(-a * b) for a, b in zip(Powers_5, Price_real)])
print('- Session 5 (re-scheduled 8-24h of D  & D1 with 4% error): {} €'.format(round(CashFlow_5_real, 2)))

# Session 6
Powers_6, SOC_6 = scheduler(SOC_3[13], Price_6[13:len(Price_6)], Batt_Emax, Batt_Pmax, Batt_Efficiency, 35)
Powers_6 = Powers_5[0:13] + Powers_6
SOC_6 = SOC_5[0:13] + SOC_6
CashFlow_6_real = sum([(-a * b) for a, b in zip(Powers_6, Price_real)])
print('- Session 6 (re-scheduled 13-24h of D & D1 with 2% error): {} €'.format(round(CashFlow_6_real, 2)))

# Session 7
Powers_7, SOC_7 = scheduler(SOC_i, Price_7, Batt_Emax, Batt_Pmax, Batt_Efficiency, len(Price_7))
CashFlow_7 = sum([(-a * b) for a, b in zip(Powers_7, Price_7)])
print('- Cash flow  with no errors woul\'ve been {} €'.format(round(CashFlow_7, 2)))

# Separate
Powers_D, SOC_D = scheduler(SOC_i, Price_real[0:24], Batt_Emax, Batt_Pmax, Batt_Efficiency, 24)
Powers_D1, SOC_D1 = scheduler(SOC_i, Price_real[24:48], Batt_Emax, Batt_Pmax, Batt_Efficiency, 24)
CashFlow_sep = sum([(-a * b) for a, b in zip(Powers_D, Price_real[0:24])]) + sum([(-a * b) for a, b in zip(Powers_D1, Price_real[24:48])])
print(SOC_D + SOC_D1)
print(SOC_7)

print('- Cash flow operating in both diary markets woul\'ve been {} €'.format(round(CashFlow_sep, 2)))

# -- Plots --
fig = plt.figure()
x = np.arange(48)
# Session 0
ax = fig.add_subplot(9, 2, 1)
plt.bar(x, Powers_0, color='g', zorder=2)
plt.ylabel('Session 0')
plt.title('Schedule (MW)')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
ax = fig.add_subplot(9, 2, 2)
plt.plot(Price_0, 'b')
plt.title('Prices (€/MWh)')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
# Session 1
ax = fig.add_subplot(9, 2, 3)
plt.bar(x, Powers_1, color='g', zorder=2)
plt.ylabel('Session 1')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
ax = fig.add_subplot(9, 2, 4)
plt.plot(Price_1, 'b')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
# Session 2
ax = fig.add_subplot(9, 2, 5)
plt.bar(x, Powers_2, color='g', zorder=2)
plt.ylabel('Session 2')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
ax = fig.add_subplot(9, 2, 6)
plt.plot(Price_2, 'b')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
# Session 3
ax = fig.add_subplot(9, 2, 7)
plt.bar(x, Powers_3, color='g', zorder=2)
plt.ylabel('Session 3')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
ax = fig.add_subplot(9, 2, 8)
plt.plot(Price_3, 'b')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
# Session 4
ax = fig.add_subplot(9, 2, 9)
plt.bar(x, Powers_4, color='g', zorder=2)
plt.ylabel('Session 4')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
ax = fig.add_subplot(9, 2, 10)
plt.plot(Price_4, 'b')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
# Session 5
ax = fig.add_subplot(9, 2, 11)
plt.bar(x, Powers_5, color='g', zorder=2)
plt.ylabel('Session 5')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
ax = fig.add_subplot(9, 2, 12)
plt.plot(Price_5, 'b')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
# Session 6
ax = fig.add_subplot(9, 2, 13)
plt.bar(x, Powers_6, color='g', zorder=2)
plt.ylabel('Session 6')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
ax = fig.add_subplot(9, 2, 14)
plt.plot(Price_6, 'b')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
# Session 7
ax = fig.add_subplot(9, 2, 15)
plt.bar(x, Powers_7, color='g', zorder=2)
plt.ylabel('Real schedule')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
ax = fig.add_subplot(9, 2, 16)
plt.plot(Price_7, 'b')
plt.xticks(color='w')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
# Separate
ax = fig.add_subplot(9, 2, 17)
plt.bar(x, Powers_D + Powers_D1, color='g', zorder=2)
plt.ylabel('Separate schedule')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()
ax = fig.add_subplot(9, 2, 18)
plt.plot(Price_7, 'b')
ax.set_xticks(x, 24)
ax.set_xlim(0, 48)
ax.grid()

# Launching the plot
plt.show()

