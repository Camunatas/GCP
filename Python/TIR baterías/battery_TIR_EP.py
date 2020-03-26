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
	return SOC, P_output


# Obtaining IRR for different battery powers
Powers = []
Capacities = []
TIRs = []
for Batt_Pmax in np.arange(1, 10, 0.5):
	for Batt_Emax in np.arange(1, 10, 0.5):
		Cap_cost = 50000 * Batt_Pmax + 200000 * Batt_Emax
		SOC, P_output = daily(SOC_i, Price, Batt_Emax, Batt_Pmax, Batt_Efficiency)
		P_output_total = sum([abs(ele) for ele in P_output])
		OM_cost = 10 * P_output_total
		SOC = [i * (100 // Batt_Emax) for i in SOC]
		SOC.append(0)
		if SOC[-1] == 0:
			P_output[-1] = 0
		FC = sum([(-a * b)*365*25 - OM_cost for a, b in zip(P_output, Price)])
		IRR = npf.irr([-Cap_cost, FC])
		Capacities.append(Batt_Emax)
		Powers.append(Batt_Pmax)
		TIRs.append(IRR)

optP = Powers[TIRs.index(max(TIRs))]
optE = Capacities[TIRs.index(max(TIRs))]
# Creating grid for surface plotting with scattered data
fig = pylab.figure()
X = Powers
Y = Capacities
Z = TIRs

ax = fig.gca(projection='3d')
ax.plot_trisurf(X, Y, Z, color='g', alpha=0.75)
ax.text2D(0.05, 0.95, 'Optimum power is {} MW, optimum capacity is {} MWh'.format(optP, optE), transform=ax.transAxes)
ax.scatter(optP, optE, max(TIRs), s=100, color='r', alpha=1)

# Plot labels
ax.set_xlabel('Power (MW)')
ax.set_ylabel('Capacity (MWh)')
ax.set_zlabel('IRR (%)')

pylab.show()