import numpy as np
from scenredpy import Class_scenred

sr_instance = Class_scenred()   # Creates sr_instance

sr_instance.import_data("test_data.h5") # Loads data
sr_instance.prepare_data()  # Prepares data

#sr_instance.scenario_reduction(dist_type="cityblock", fix_node=1, tol_node=np.linspace(1,24, 24)) #fix_prob_tol
sr_instance.scenario_reduction(dist_type="cityblock",fix_prob=1, tol_prob=np.linspace(0, 0.2, 24))  # Runs reductor

sr_instance.draw_red_scenario()     # Plots reduced scenarios

sr_instance.sort_result()   # Prints scenario reduction process







