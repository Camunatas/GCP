DODs = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
cycles = [15000,7000, 3300, 2050, 1475, 1150, 950, 760, 675, 580, 500];

figure;
hold on
plot(DODs, cycles,'LineWidth',2)
grid on
grid minor
ylim([0 15500])
xlim([0 110])
xlabel('DOD')
ylabel('Cycles')
plot(5,15000, 'r*')
text(6,15100, '15000')
plot(10, 7000, 'r*')
text(11,7100, '7000')
plot(20, 3300, 'r*')
text(21,3600, '3300')
plot(30, 2050, 'r*')
text(32,2400, '2050')
plot(40, 1475, 'r*')
text(35,1050, '1475')
plot(50, 1150, 'r*')
text(45,1700, '1150')
plot(60, 950, 'r*')
text(57.5,1500, '950')
plot(70, 760, 'r*')
text(67.5,1300, '760')
plot(80, 675, 'r*')
text(77.5,1200, '675')
plot(90, 580, 'r*')
text(87.5,1150, '580')
plot(100, 500, 'r*')
text(97.5,1100, '500')
hold off