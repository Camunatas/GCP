% Aging mechanism emulator
% Author: Pedro Luis Camuñas García-Miguel
clear 
clc
%% Cycling parameters
T_amb = 25;               % [°] Ambient temperature

%% Loading data from NASA database
battery_ID = 5;
battery = num2str(battery_ID);

% Loading battery
if battery_ID < 10
    BATT = load(['NASA/B000' battery '.mat']);
else
    BATT = load(['NASA/B00' battery '.mat']);
end

% Loading capacity data
realcaps = [];
realtemps = [];
for i = 1:size(eval(['BATT.B000' battery '.cycle']),2)
    if strcmp(eval(['BATT.B000' battery '.cycle(' int2str(i) ').type']),'discharge') == 1
        realcaps = vertcat(realcaps, eval(['BATT.B000' battery '.cycle(' int2str(i) ').data.Capacity']));
        realtemps = vertcat(realtemps, mean(eval(['BATT.B000' battery '.cycle(' int2str(i) ').data.Temperature_measured'])));
    else
    end
end

%% Theoretical Battery Ager
Cap_min = realcaps(1);
Cycles = 0;
cycle_count = 0;
Cap = realcaps(1);
NewCap = realcaps(1);

while cycle_count < size(realtemps,1)
	cycle_count = cycle_count+1;
    Cap_fade = 2*0.01*0.000249*exp(0.02717*(298+realtemps(cycle_count)))*0.02982*(1^0.4904)*1^0.5;
	NewCap = NewCap*(1-Cap_fade);
	Cap = horzcat(Cap,NewCap);
	Cycles = horzcat(Cycles,cycle_count);
end

%% Plotter
figure(1)
subplot(1,3,1)
plot(Cycles,Cap)
title('Theorical capacity fade')
xlabel('Cycles')
ylabel('Capacity (Ah)')
xlim([0 size(Cycles,2)])
ylim([min(Cap) max(Cap)])
subplot(1,3,2)
plot(realcaps)
title('Real capacity fade')
xlabel('Cycles')
ylabel('Capacity (Ah)')
xlim([0 size(realcaps,1)])
ylim([min(realcaps) max(realcaps)])
subplot(1,3,3)
plot(realtemps)
title('Real cell temperature')
xlabel('Cycles')
ylabel('Temperature (° C)')


%% Display
disp('Theoretical aging results:');
fprintf('\t - Capacity fades from %.2f to %.5f Ah.\n',realcaps(1), NewCap);
fprintf('\t - EOL met at %d cycles.\n',cycle_count);
disp('Real battery aging results:');
fprintf('\t - Capacity fades from %.2f to %.5f Ah.\n',realcaps(1), realcaps(size(realcaps,1)));
fprintf('\t - EOL met at %d cycles.\n',size(realcaps,1));