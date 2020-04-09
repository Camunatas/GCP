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


# Obtaining optimal by brute force
# Powers = []
# Capacities = []
# TIRs = []
# for Batt_Pmax in np.arange(1, 10, 0.5):
# 	for Batt_E in np.arange(1, 10, 0.5):
# 		SOC, P_output, Capital_cost, E_output, OM_cost, earning, FC, IRR = \
# 		daily(SOC_i, Price, Batt_E, Batt_Pmax, Batt_Efficiency, Project_length)
# 		Capacities.append(Batt_E)
# 		Powers.append(Batt_Pmax)
# 		TIRs.append(IRR)

# -- Obtaining optimal by perturb and observe --
# Initializing vectors
IRRs = []
Capacities = []
Powers = []
# Algorithm step
delta = 0.5

# Applying P&O algorithm for a range of energies
Batt_E = 5
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
for Batt_E in np.arange(1, 10, 0.5):
	while enabler == 1:
		if IRR_local[-1] - IRR_local[-2] > 0:
			print(1)
			if Powers_local[-1] - Powers_local[-2] > 0:
				print(1.1)
				Power = Power + delta
				IRR = daily(SOC_i, Price, Batt_E, Power, Batt_Efficiency, Project_length)
				Powers_local.append(Power)
				IRR_local.append(IRR)
			if Powers_local[-1] - Powers_local[-2] < 0:
				print(1.2)
				Power = Power - delta
				IRR = daily(SOC_i, Price, Batt_E, Power, Batt_Efficiency, Project_length)
				Powers_local.append(Power)
				IRR_local.append(IRR)
		elif IRR_local[-1] - IRR_local[-2] < 0:
			print(2)
			if Powers_local[-1] - Powers_local[-2] > 0:
				print(2.1)
				Power = Power - delta
				IRR = daily(SOC_i, Price, Batt_E, Power, Batt_Efficiency, Project_length)
				Powers_local.append(Power)
				IRR_local.append(IRR)
			elif Powers_local[-1] - Powers_local[-2] < 0:
				print(2.2)
				Power = Power + delta
				IRR = daily(SOC_i, Price, Batt_E, Power, Batt_Efficiency, Project_length)
				Powers_local.append(Power)
				IRR_local.append(IRR)
		if len(IRR_local) > 2:
				print(2.3)
				if IRR_local[-1] < IRR_local[-2] and IRR_local[-2] > IRR_local[-3]:
					enabler = 0
	Capacities.append(Batt_E)
	Powers.append(Powers_local[-2])
	IRRs.append(IRR_local[-2])

print('Optimal IRR is {} at {} MW'.format(max(IRRs), Powers_local[-2]))


# Displaying results for optimum
# Capital_cost = 50000 * Batt_Pmax + 200000 * Batt_Emax
# SOC, P_output = daily(SOC_i, Price, Batt_Emax, Batt_Pmax, Batt_Efficiency)
# E_output_total = sum([abs(ele) for ele in P_output])
# OM_cost = 10 * E_output_total
# SOC = [i * (100 // Batt_Emax) for i in SOC]
# SOC.append(0)
# if SOC[-1] == 0:
# 	P_output[-1] = 0
# earning = sum([(-a * b) for a, b in zip(P_output, Price)]) * improve_rate
# FC = (earning - OM_cost) * project_length
# IRR = npf.irr([-Capital_cost, FC]) * 100


# print('Results for optimal point which is found at {} MW, {} MWh, ({})h'.format(optP, optE, round(optE/optP, 2)))
# print('Capital cost is {}€'.format(int(Capital_cost)))
# print('It moves {} MW a day, {} MW in total'.format(round(E_output, 2), round(E_output*Project_length, 2)))
# print('O&M costs reach {}€'.format(round(abs(OM_cost), 2)))
# print('The project earnings are {}€'.format(round(earning, 2)))
# print('Cash flux is {}€'.format(round(FC, 2)))
# print('Project total balance is {}€'.format(round(FC-Capital_cost, 2)))
# print('TIR is {}%'.format(round(IRR, 2)))
#
# # Creating grid for surface plotting with scattered data
# fig = pylab.figure()
# X = Powers
# Y = Capacities
# Z = TIRs
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(X, Y, Z, color='g', alpha=0.75)
# ax.text2D(0.05, 0.95, 'Optimum power is {} MW, optimum capacity is {} MWh, ({})h'.format(optP, optE, round(optE/optP, 2)), transform=ax.transAxes)
# ax.scatter(optP, optE, max(TIRs), s=100, color='r', alpha=1)
#
# # Plot labels
# ax.set_xlabel('Power (MW)')
# ax.set_ylabel('Capacity (MWh)')
# ax.set_zlabel('IRR (%)')

pylab.show()
