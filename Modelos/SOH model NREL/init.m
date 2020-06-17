%% Script description
% Emulation of degradation model of paper Life Prediction Model for Grid- 
% Connected Li-ion Battery Energy Storage System- From Kandler Smith, 
% Aron Saxon, Matthew Keyser, and Blake Lundstrom
% Author: Pedro Luis Camuñas
% Date: 30/05/2020
% This script initializes the variables of the model 
%%
clear 
clc
%% Simulation parameteres
Tsim_d = 300;               % [days] Simulation time
Tsim = 24*3600*Tsim_d;      % [s] Simulation time
%% Reference constants
T_ref = 298.15;             % [K] Reference temperature
V_ref = 3.7;                % [V] Reference voltage
U_neg_ref= 0.08;            % [V] Reference reverse polarization voltage
F = 96485;                  % [A*s/mol] Faraday constant
R_ug = 8.314;               % [J/(K*mol)] Universal gas constant
%% Capacity model fitted parameters
% Positive electrode-site-limited capacity parameters
d_0_ref = 75.1;             % [Ah]
E_a_d_0_1 = 4126;           % [mol^-1]
E_a_d_0_2 = 9.752e6;        % [J/mol]
d_3 = 0.46;                 % [Ah]

% Li-limited capacity parameters
b_0 = 1.07;
b_1_ref = 3.503e-3;         % [sqrt(days)]
E_a_b_1 = 35392;            % [J/mol]
alfa_b_1 = -1;
gamma_b_1 = 2.472;
beta_b1 = 2.157;

b2_ref = 1.541e-5;
E_a_b2 = -42800;            % [J/mol]

b3_ref = 2.805e-2;
E_a_b3 = 42800;             % [J/mol]
alfa_b_3 = 0.0066;
theta = 0.135;
tau_b_3 = 5;

% Negative electrode-site-limited capacity parameters
E_a_c0 = 2224;              % [J/mol] J mol^1 (fit)
c_0_ref = 75.64;            % [Ah]
c_2_ref = 3.9193e-3;        % [Ah/cycle]
E_a_c2 = -48260;            % [J/mol]
beta_c2 = 4.54;