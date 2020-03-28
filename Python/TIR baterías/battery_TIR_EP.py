# --- IRR of a Battery Performing Arbitrage for different Powers and Energies---
# Author: Pedro Luis Camuñas
# Date: 21/03/2020

from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import numpy_financial as npf
import numpy as np
import pandas as pd

# -- Simulation parameters --
Batt_Efficiency = 0.9               # Rated battery efficiency
Batt_ncycles = 4500					# Rated lifetime in cycles
project_length = 25*365				# Project duration (days)
# -- Data initialization --
SOC_i = 0                           # [%] Initial battery SOC

# -- Data importing --
data = pd.read_excel('Prices.xlsx', sheet_name='Prices', nrows=200)

# -- Generating price prediction
Price = list(data['Price'][0:25])

# -- Daily market --
def daily(initial_SOC, energy_price, batt_capacity, batt_maxpower, batt_efficiency):
	# Model initialization
	model = ConcreteModel()
	model.time = range(24)
	model.time2 = range(1, 24)
	model.time3 = range(25)
	model.SOC = Var(model.time3, bounds=(0, batt_capacity))         # Battery SOC
	model.not_charging = Var(model.time, domain=Binary)             # Charge verifier
	model.not_discharging = Var(model.time, domain=Binary)          # Discharge verifier
	model.ESS_C = Var(model.time, bounds=(0, batt_maxpower))        # Energy being charged
	model.ESS_D = Var(model.time, bounds=(0, batt_maxpower))        # Energy being discharged

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

	# Objective Function: Maximize profitability
	model.obj = Objective(
		expr=sum(((energy_price[t1] * (model.ESS_D[t1] - model.ESS_C[t1]))
		          for t1 in model.time)), sense=maximize)

	# Applying the solver
	opt = SolverFactory('cbc')
	opt.solve(model)
	# model.pprint()

	# Extracting data from model
	SOC = [model.SOC[t1]() for t1 in model.time]
	P_output = [-model.ESS_D[t1]() + model.ESS_C[t1]() for t1 in model.time]
	Cap_fade = [0.00024 * np.exp(0.02717*298)*0.02982 * np.power(((model.ESS_D[t1]() + model.ESS_C[t1]()) / batt_capacity), 0.4904) * np.power(0.5, 0.5) for t1 in model.time]
	return SOC, P_output, Cap_fade


# Obtaining IRR for different battery powers
Powers = []
Capacities = []
TIRs = []
for Batt_Pmax in np.arange(1, 10, 0.5):
	for Batt_Emax in np.arange(1, 10, 0.5):
		Capital_cost = 50000 * Batt_Pmax + 200000 * Batt_Emax
		SOC, P_output, Cap_fade = daily(SOC_i, Price, Batt_Emax, Batt_Pmax, Batt_Efficiency)
		E_output_total = sum([abs(ele) for ele in P_output])
		n_cycles = (E_output_total / (Batt_Emax*2)) * project_length
		Cap_loss = round(sum(Cap_fade) * project_length, 2)
		n_batteries = n_cycles / Batt_ncycles
		replacements = Cap_loss // 100
		OM_cost = Capital_cost*(replacements - 1)
		SOC = [i * (100 // Batt_Emax) for i in SOC]
		SOC.append(0)
		if SOC[-1] == 0:
			P_output[-1] = 0
		earning = sum([(-a * b) * project_length for a, b in zip(P_output, Price)])
		FC = earning - OM_cost
		IRR = npf.irr([-Capital_cost, FC])
		Capacities.append(Batt_Emax)
		Powers.append(Batt_Pmax)
		TIRs.append(IRR)

# Obtaining optimum
optP = Powers[TIRs.index(max(TIRs))]
optE = Capacities[TIRs.index(max(TIRs))]

optP = 3
optE = 6

# Displaying results for optimum
Capital_cost = 50000 * optP + 200000 * optE
SOC, P_output, Cap_fade = daily(SOC_i, Price, optP, optE, Batt_Efficiency)
E_output_total = sum([abs(ele) for ele in P_output])
n_cycles = (E_output_total / (optE*2)) * project_length
Cap_loss = round(sum(Cap_fade) * project_length, 2)
n_batteries = n_cycles / Batt_ncycles
replacements = Cap_loss // 100
OM_cost = Capital_cost*(replacements - 1)
SOC = [i * (100 // optE) for i in SOC]
SOC.append(0)
if SOC[-1] == 0:
	P_output[-1] = 0
earning = sum([(-a * b) * project_length for a, b in zip(P_output, Price)])
FC = earning - OM_cost
IRR = npf.irr([-Capital_cost, FC])

print('Results for {} MW, {} MWh, ({})h'.format(optP, optE, round(optE/optP, 2)))
print('Capital cost is {}€'.format(int(Capital_cost)))
print('It moves {} MW a day, {} MW in total'.format(round(E_output_total, 2), round(E_output_total*project_length, 2)))
print('It performs {} cycles, which can be considered as costing ({}) batteries'.format(round(n_cycles, 2), round(n_batteries, 2)))
print('Theorically, capacity fades a {}%, costing {} batteries'.format(round(Cap_loss, 2), int(replacements)))
print('O&M costs reach {}€'.format(round(abs(OM_cost), 2)))
print('The project earnings are {}€'.format(round(earning, 2)))
print('Cash flux is {}€'.format(round(FC, 2)))
print('Project total balance is {}€'.format(round(FC-Capital_cost, 2)))
print('TIR is {}%'.format(round(IRR, 2)))

# Creating grid for surface plotting with scattered data
fig = pylab.figure()
X = Powers
Y = Capacities
Z = TIRs
ax = fig.gca(projection='3d')
ax.plot_trisurf(X, Y, Z, color='g', alpha=0.75)
ax.text2D(0.05, 0.95, 'Optimum power is {} MW, optimum capacity is {} MWh, ({})h'.format(optP, optE, round(optE/optP, 2)), transform=ax.transAxes)
ax.scatter(optP, optE, max(TIRs), s=100, color='r', alpha=1)

# Plot labels
ax.set_xlabel('Power (MW)')
ax.set_ylabel('Capacity (MWh)')
ax.set_zlabel('IRR (%)')

pylab.show()