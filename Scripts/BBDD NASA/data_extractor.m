% Script for extracting data from NASA database
% Author: Pedro Luis Camuñas García-Miguel
clear 
clc

%% Loading data from NASA database
capsexcel = [];
for battery_ID = 5:56
    % Loading battery
    if battery_ID < 10
        filename = strcat('NASA/B000',num2str(battery_ID),'.mat');
        battstruct = 'B000';
    else
        filename = strcat('NASA/B00', num2str(battery_ID),'.mat');
        battstruct = 'B00';
    end
    if isfile(filename)
        BATT = load(filename);
        % Loading capacity data
        realcaps = [];
        realtemps = [];
        for i = 1:size(eval(['BATT.' battstruct num2str(battery_ID) '.cycle']),2)
            if strcmp(eval(['BATT.' battstruct num2str(battery_ID) '.cycle(' int2str(i) ').type']),'discharge') == 1
                realcaps = vertcat(realcaps, eval(['BATT.' battstruct num2str(battery_ID) '.cycle(' int2str(i) ').data.Capacity']));
                realtemps = vertcat(realtemps, mean(eval(['BATT.' battstruct num2str(battery_ID) '.cycle(' int2str(i) ').data.Temperature_measured'])));
            else
            end
        end
    else
    end
    xlswrite(num2str(battery_ID), realcaps);
end