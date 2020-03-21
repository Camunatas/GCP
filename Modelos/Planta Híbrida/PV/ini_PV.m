%% PV panel parameters
clear all
clc
%% Parameters
Pv.Voc = 64.2;                                                      % [V] Open circuit voltage 
Pv.Isc = 5.96;                                                      % [A] Short-circuit current 
Pv.Pnom = 305;                                                      % [W] Maximum power 
Pv.Ns = 96;                                                         % Cells per module
Pv.Rs = 0.37152;                                                    % [Ohms] Series resistance 
Pv.Tsc = 0.061745;                                                  % [%/ºC] Temperature coeficient for Isc 
Pv.A = 0.94504;                                                     % Diode ideality factor 
