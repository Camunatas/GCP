%% Simulink model of a Hybrid plant
clear all
clc
%% Simulation parameters
t_simh = 24;                            % Simlation time (h)
t_sim = 3600*t_simh;                     % Simulation time (s)

%% Plant parameters
% Environment
Amb.ro = 1.225;                         % [kg/m^3] Air density 
% Wind turbine
Wt.radio = 42;                          % [m] Blades Radio 
Wt.n_g = 0.97;                          % Generator Efficiency
Wt.n_b = 0.95;                          % Gearbox Efficiency
Wt.Pnom = 2.4;                          % [MW] Nominal Power

Wt.C_1= 0.73;
Wt.C_2= 151;
Wt.C_3= 0.58;
Wt.C_4= 13.2;
Wt.C_5 = 18.4;
Wt.C_7= 0.002;
Wt.C_8= 2.14;

Wt.lambda_opt = 7.2;                    % Optimum Tip Speed Ratio
Wt.lambda = Wt.lambda_opt;              % Tip Speed Ratio
Wt.beta = 0;                            % Pitch Angle (°)

% PV Plant
Pv.Voc = 64.2;                          % [V] Open Circuit Voltage 
Pv.Isc = 5.96;                          % [A] Short-circuit Current 
Pv.Pnom = 305;                          % [W] Maximum Power 
Pv.Ns = 96;                             % Cells per Module
Pv.Rs = 0.37152;                        % [Ohms] Series Resistance 
Pv.Tsc = 0.061745;                      % [%/ºC] Temperature Coeficient for Isc 
Pv.A = 0.94504;                         % Diode Ideality Factor
Pv.np = 300;                            % Amount of panels

% ESS
Batt.Cap = 1.4;                         % Battery Capacity (MWh)
Batt.Pmax = 0.7;                        % Battery Rated Power (MW)
Batt.Soc_i = 0.1;                       % Battery Initial SOC

% Load
Lo.P = 0.001;                             % Internal Load Power (MW)

% Grid
