# --- Optimization of the sizing of a Standalone Battery doing arbitrage by P&O algorithm---
# Author: Pedro Luis Camuñas
# Date: 09/04/2020

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
Project_length = 25*365				# Project duration (days)
# -- Data initialization --
SOC_i = 0                           # [%] Initial battery SOC

# -- Data importing --
data = pd.read_excel('Prices.xlsx', sheet_name='Prices', nrows=200)

# -- Generating price prediction
Price = list(data['Price'][0:25])

# -- Daily market --
def daily(initial_SOC, energy_price, batt_capacity, batt_maxpower, batt_efficiency, project_length):
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
	SOC = [i * (100 // batt_capacity) for i in SOC]
	SOC.append(0)
	if SOC[-1] == 0:
		P_output[-1] = 0
	E_output = sum([abs(ele) for ele in P_output])
	OM_cost = 10 * E_output
	Capital_cost = 50000 * batt_maxpower + 200000 * batt_capacity
	improve_rate = 10
	earning = sum([(-a * b) for a, b in zip(P_output, Price)]) * improve_rate
	FC = (earning - OM_cost) * project_length
	IRR = npf.irr([-Capital_cost, FC]) * 100
	return IRR

# -- Obtaining optimal by perturb and observe --
# Initializing vectors
IRRs = []
Capacities = []
Powers = []
# Algorithm step
delta = 0.5
loopcount = 0
# Applying P&O algorithm for a range of energies
for Batt_E in np.arange(1, 10, 0.5):
	IRR_local = [0]				# Local IRR vector
	Powers_local = [0]			# Local powers vector
	Power = 5
	IRR = daily(SOC_i, Price, Batt_E, Power, Batt_Efficiency, Project_length)
	Powers_local.append(Power)
	IRR_local.append(IRR)
	Power = Power + delta
	IRR = daily(SOC_i, Price, Batt_E, Power, Batt_Efficiency, Project_length)
	Powers_local.append(Power)
	IRR_local.append(IRR)
	enabler = 1
	while enabler == 1:
		if IRR_local[-1] - IRR_local[-2] > 0:
			if Powers_local[-1] - Powers_local[-2] > 0:
				Power = Power + delta
				IRR = daily(SOC_i, Price, Batt_E, Power, Batt_Efficiency, Project_length)
				Powers_local.append(Power)
				IRR_local.append(IRR)
			if Powers_local[-1] - Powers_local[-2] < 0:
				Power = Power - delta
				IRR = daily(SOC_i, Price, Batt_E, Power, Batt_Efficiency, Project_length)
				Powers_local.append(Power)
				IRR_local.append(IRR)
		elif IRR_local[-1] - IRR_local[-2] < 0:
			if Powers_local[-1] - Powers_local[-2] > 0:
				Power = Power - delta
				IRR = daily(SOC_i, Price, Batt_E, Power, Batt_Efficiency, Project_length)
				Powers_local.append(Power)
				IRR_local.append(IRR)
			elif Powers_local[-1] - Powers_local[-2] < 0:
				Power = Power + delta
				IRR = daily(SOC_i, Price, Batt_E, Power, Batt_Efficiency, Project_length)
				Powers_local.append(Power)
				IRR_local.append(IRR)
		if len(IRR_local) > 2:
				if IRR_local[-1] < IRR_local[-2] and IRR_local[-2] > IRR_local[-3]:
					enabler = 0
		loopcount = loopcount + 1
	Capacities.append(Batt_E)
	Powers.append(Powers_local[-2])
	IRRs.append(IRR_local[-2])

# Extracting optimal and displaying results
optIRR = max(IRRs)
optP = Powers[IRRs.index(max(IRRs))]
optE = Capacities[IRRs.index(max(IRRs))]
print('Optimal IRR is {} at {} MW and {} MWh ({}) hours'.format(optIRR, optP, optE, optE/optP))
print('Optimization process took {} simulations'.format(loopcount))

# Obtaining IRR for different battery powers
Powers_bruteforce = []
Capacities_bruteforce = []
IRRs_bruteforce = []
largeloopcounter = 0
for Batt_Pmax in np.arange(1, 10, 0.5):
	for Batt_Emax in np.arange(1, 10, 0.5):
		IRR = daily(SOC_i, Price, Batt_Emax, Batt_Pmax, Batt_Efficiency, Project_length)
		Capacities_bruteforce.append(Batt_Emax)
		Powers_bruteforce.append(Batt_Pmax)
		IRRs_bruteforce.append(IRR)
		largeloopcounter = largeloopcounter + 1

optIRR_bruteforce = max(IRRs_bruteforce)
optP_bruteforce = Powers_bruteforce[IRRs_bruteforce.index(optIRR_bruteforce)]
optE_bruteforce = Capacities_bruteforce[IRRs_bruteforce.index(optIRR_bruteforce)]

print('Optimal IRR found by brute force is {} at {} MW and {} MWh ({}) hours'.format(optIRR_bruteforce, optP_bruteforce, optE_bruteforce, optE_bruteforce/optP_bruteforce))
print('Brute force took {} simulations'.format(largeloopcounter))

# Creating grid for surface plotting with scattered data
fig = pylab.figure()
X = Powers_bruteforce
Y = Capacities_bruteforce
Z = IRRs_bruteforce
ax = fig.gca(projection='3d')
ax.plot_trisurf(X, Y, Z, color='g', alpha=0.75)
ax.text2D(0.05, 0.95, 'Optimum power is {} MW, optimum capacity is {} MWh, ({})h'.format(optP_bruteforce, optE, round(optE_bruteforce/optP_bruteforce, 2)), transform=ax.transAxes)
ax.scatter(optP_bruteforce, optE_bruteforce, optIRR_bruteforce, s=100, color='r', alpha=1)
ax.scatter(optP, optE, optIRR, s=100, color='b', alpha=1)

# Plot labels
ax.set_xlabel('Power (MW)')
ax.set_ylabel('Capacity (MWh)')
ax.set_zlabel('IRR (%)')

pylab.show()