AQdata = load("AQdata.mat","AQdata");
data = AQdata.AQdata.data;
headers = AQdata.AQdata.hdr;
original_NO2 = data(:,8);
original_O3 = data(:,9);

figure(1)
plot(original_NO2)
title("Original NO2 data")
ylabel("NO2 concentration")
xlabel("Sample")
figure(2)
plot(original_O3)
title("Original O3 data")
ylabel("O3 concentration")
xlabel("Sample")

[std_NO2, stat_std_NO2] = dataprep(original_NO2,'std');
[mms_NO2, stat_mms_NO2] = dataprep(original_NO2,'mms');

[std_O3, stat_std_O3] = dataprep(original_O3,'std');
[mms_O3, stat_mms_O3] = dataprep(original_O3,'mms');


figure("Name","Method comparison for NO2")

subplot(3,1,1)
plot(original_NO2,"color","K","LineWidth",0.1)
title("Original NO2 data")
ylabel("NO2 concentration")
xlabel("Sample")
grid on
ylim([-1 1000])

subplot(3,1,2)
plot(std_NO2,"color","b")
title("std scaled NO2 data")
ylabel("NO2 concentration")
xlabel("Sample")
grid on
ylim([-1 5])

subplot(3,1,3)
plot(mms_NO2,"color","r")
title("mms scaled NO2 data")
ylabel("NO2 concentration")
xlabel("Sample")
ylim([-1 1])
grid on
figure("Name","Method comparison for O3")

subplot(3,1,1)
plot(original_O3,"color","K")
title("Original O3 data")
ylabel("O3 concentration")
xlabel("Sample")
grid on

subplot(3,1,2)
plot(std_O3,"color","b")
title("std scaled O3 data")
ylabel("O3 concentration")
xlabel("Sample")
grid on


subplot(3,1,3)
plot(mms_O3,"color","r")
title("mms scaled O3 data")
ylabel("O3 concentration")
xlabel("Sample")
grid on
%%
%This section plots the data as a whole after the scaling process.

std_data = data;
std_data(:,8) = std_NO2;
std_data(:,9) = std_O3;

mms_data = data;
mms_data(:,8) = mms_NO2;
mms_data(:,9) = mms_O3;

figure("Name","Original Data")
plot(data)
title("Original data")

figure("Name","Std scaled Data")
plot(std_data)
title("Std Scaled data")

figure("Name","Mms Scaled Data")
plot(mms_data)
title("Mms Scaled data")