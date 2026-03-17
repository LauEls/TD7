%% Final Evaluation
clear all;
close all;

% options_2.color_area = [0.8500 0.3250 0.0980];
% options_2.color_line = [0.8500 0.3250 0.0980];
% 
% 
% options_3.color_area = [0.9290 0.6940 0.1250];
% options_3.color_line = [0.9290 0.6940 0.1250];
% 
% 
% options_4.color_area = [0.4660 0.6740 0.1880];
% options_4.color_line = [0.4660 0.6740 0.1880];
% 
% 
% options_5.color_area = [0.4940 0.1840 0.5560];
% options_5.color_line = [0.4940 0.1840 0.5560];


% x = categorical{}
figure('Position',[0 0 1920 1440]);
hold on;

bar_categories = {'TD7+Demo','TD7', 'BC', 'Demo Replay'};
% bar_categories = {'1','2', '3', '4'};
X = categorical(bar_categories);
X = reordercats(X,bar_categories);
Y = [0.96 0.84; 0.0 0.29; 0.74 0.62; 0.28 0.37];
% X = reordercats(X,{'TD7','TD7+Demo', 'BC', 'Demo Replay'});
% Y = [10 21 33 52];
b = bar(X,Y, 'grouped','GroupWidth',0.9, 'BarWidth',0.9);

% axis tight;


% Y = [10 21 33 52];
% b.LineWidth = line_width;
b(1).FaceAlpha = 0.9;
b(1).FaceColor = 'flat';
b(2).FaceAlpha = 0.9;
b(2).FaceColor = 'flat';
% b.CData(1,:) = options_2.color_line;
% b.CData(2,:) = options_6.color_line;
b(2).CData = [0.8500 0.3250 0.0980];
b(1).CData = [0.4660 0.6740 0.1880];
% b.CData(3,:) = options_4.color_line;
% b.CData(4,:) = options_5.color_line;

b(1).Labels = b(1).YData;
b(2).Labels = b(2).YData;

% e = errorbar(X, bar_data, error_data);
% e.Color = 'black';
% e.LineStyle = 'none';
% e.LineWidth = line_width;
% e.CapSize = 40;

%xlabel('Episodes in Demonstration Buffer')
% ylabel('Mean Reward and Variance')
ylim([0.0 1.0])

% set(gca,'FontSize',18)
fontsize(gcf, 30, "points")
set(gca,'FontSize',55)
hold off;

%% Final Evaluation v2
clear all;
close all;

figure('Position',[0 0 1920 1440]);
hold on;

bar_categories = {'Success Rate','Mean Reward'};
X = categorical(bar_categories);
X = reordercats(X,bar_categories);
% Y = [0.96 0.84; 0.0 0.29; 0.74 0.62; 0.28 0.37];
Y = [0.96 0.00 0.74 0.28; 0.84 0.29 0.62 0.37];
% X = reordercats(X,{'TD7','TD7+Demo', 'BC', 'Demo Replay'});
% Y = [10 21 33 52];
b = bar(X,Y, 'grouped','GroupWidth',0.6, 'BarWidth',1.0, 'FaceAlpha',0.9, 'FaceColor','flat');

b(1).CData = [0.8500 0.3250 0.0980];
b(2).CData = [0.9290 0.6940 0.1250];
b(3).CData = [0.4660 0.6740 0.1880];
b(4).CData = [0.4940 0.1840 0.5560];

b(1).Labels = b(1).YData;
b(2).Labels = b(2).YData;
b(3).Labels = b(3).YData;
b(4).Labels = b(4).YData;

% ax = gca;
% 
% % Get current x-axis limits
% currentXlim = xlim(ax);
% 
% % Set new x-axis limits to add a margin at the beginning (e.g., a margin of 0.5 units)
% newXlimMin = currentXlim(1) - 0.5; % Subtract a small value
% newXlimMax = currentXlim(2); % Keep the max limit the same
% xlim(ax, [newXlimMin, newXlimMax]);


%xlabel('Episodes in Demonstration Buffer')
% ylabel('Mean Reward and Variance')
ylim([0.0 1.0])

% set(gca,'FontSize',18)
fontsize(gcf, 28, "points")
set(gca,'FontSize',55)
hold off;


%% Comparing offline learning with different datasets
clear all;
close all;

td7_file_base = "door_mirror/gh360/osc_pose/offline/";
runs = 5;
episode_length = 500;
evaluation_frequency = 10;
maximum_timesteps = episode_length * 200;

offline_expert_raw = cell(1, runs);
offline_expert_mean = cell(1,runs);
offline_random_expert_raw = cell(1, runs);
offline_random_expert_mean = cell(1,runs);
offline_gradual_random_expert_raw = cell(1, runs);
offline_gradual_random_expert_mean = cell(1,runs);

for i=1:runs
    offline_expert_raw{i} = readmatrix(td7_file_base+"v1_expert_paths/run_"+num2str(i-1)+"/results.csv");
    offline_random_expert_raw{i} = readmatrix(td7_file_base+"v2_expert_random_paths/run_"+num2str(i-1)+"/results.csv");
    offline_gradual_random_expert_raw{i} = readmatrix(td7_file_base+"v3_medium_expert_random_paths/run_"+num2str(i-1)+"/results.csv");

    offline_expert_mean{i} = mean(offline_expert_raw{i},2)/episode_length;
    offline_random_expert_mean{i} = mean(offline_random_expert_raw{i},2)/episode_length;
    offline_gradual_random_expert_mean{i} = mean(offline_gradual_random_expert_raw{i},2)/episode_length;
end

offline_expert_trans = [transpose(offline_expert_mean{1}); transpose(offline_expert_mean{2}); transpose(offline_expert_mean{3}); transpose(offline_expert_mean{4}); transpose(offline_expert_mean{5})];
offline_random_expert_trans = [transpose(offline_random_expert_mean{1}); transpose(offline_random_expert_mean{2}); transpose(offline_random_expert_mean{3}); transpose(offline_random_expert_mean{4}); transpose(offline_random_expert_mean{5})];
offline_gradual_random_expert_trans = [transpose(offline_gradual_random_expert_mean{1}); transpose(offline_gradual_random_expert_mean{2}); transpose(offline_gradual_random_expert_mean{3}); transpose(offline_gradual_random_expert_mean{4}); transpose(offline_gradual_random_expert_mean{5})];

td7_x_values = (0:evaluation_frequency*episode_length/1000:maximum_timesteps/1000);
alpha  = 0.3;
line_width = 8;
error = 'std';

options_2.color_area = [0.8500 0.3250 0.0980];
options_2.color_line = [0.8500 0.3250 0.0980];
options_2.alpha      = alpha;
options_2.line_width = line_width;
options_2.error      = error;
options_2.x_axis     = td7_x_values;

options_3.color_area = [0.9290 0.6940 0.1250];
options_3.color_line = [0.9290 0.6940 0.1250];
options_3.alpha      = alpha;
options_3.line_width = line_width;
options_3.error      = error;
options_3.x_axis     = td7_x_values;

options_4.color_area = [0.4660 0.6740 0.1880];
options_4.color_line = [0.4660 0.6740 0.1880];
options_4.alpha      = alpha;
options_4.line_width = line_width;
options_4.error      = error;
options_4.x_axis     = td7_x_values;

options_5.color_area = [0.4940 0.1840 0.5560];
options_5.color_line = [0.4940 0.1840 0.5560];
options_5.alpha      = alpha;
options_5.line_width = line_width;
options_5.error      = error;
options_5.x_axis     = td7_x_values;

options_6.color_area = [21 5 120]./255;
options_6.color_line = [21 5 120]./255;
options_6.alpha      = alpha;
options_6.line_width = line_width;
options_6.error      = error;
options_6.x_axis     = td7_x_values;

options_7.color_area = [0 255 255]./255;
options_7.color_line = [0 255 255]./255;
options_7.alpha      = alpha;
options_7.line_width = line_width;
options_7.error      = error;
options_7.x_axis     = td7_x_values;


figure('Position',[0 0 1920 1440]);
hold on
plot_areaerrorbar(offline_expert_trans, options_2);
plot_areaerrorbar(offline_random_expert_trans, options_3);
plot_areaerrorbar(offline_gradual_random_expert_trans, options_4);

% xlim([0 26])
ylim([0 1])
%lgd = legend('', 'Offline Expert', '', 'Offline Random Expert', '', 'Offline Gradual Random Expert', 'Location','best');
%lgd.NumColumns = 3;
xlabel('Time Steps (1K)','FontSize',16)
ylabel('Normalized Reward','FontSize',16)
set(gca,'FontSize',55)
%set(gca,'FontSize',18)

%title('Variable Impedance Controller Comparison')
hold off


%% Comparing influence of number of demonstrations
clear all;
close all;

td7_file_base = "door_mirror/gh360/osc_pose/online/";
runs = 5;
episode_length = 500;
evaluation_frequency = 10;
maximum_timesteps = episode_length * 200;

one_demos_raw = cell(1, runs);
one_demos_mean = cell(1,runs);
two_demos_raw = cell(1, runs);
two_demos_mean = cell(1,runs);
five_demos_raw = cell(1, runs);
five_demos_mean = cell(1,runs);
ten_demos_raw = cell(1, runs);
ten_demos_mean = cell(1,runs);
twenty_demos_raw = cell(1, runs);
twenty_demos_mean = cell(1,runs);
fifty_demos_raw = cell(1, runs);
fifty_demos_mean = cell(1,runs);

for i=1:runs
    one_demos_raw{i} = readmatrix(td7_file_base+"v16_1_demos/run_"+num2str(i-1)+"/results.csv");
    two_demos_raw{i} = readmatrix(td7_file_base+"v15_2_demos/run_"+num2str(i-1)+"/results.csv");
    five_demos_raw{i} = readmatrix(td7_file_base+"v14_5_demos/run_"+num2str(i-1)+"/results.csv");
    ten_demos_raw{i} = readmatrix(td7_file_base+"v8_10_demos/run_"+num2str(i-1)+"/results.csv");
    twenty_demos_raw{i} = readmatrix(td7_file_base+"v9_20_demos/run_"+num2str(i-1)+"/results.csv");
    fifty_demos_raw{i} = readmatrix(td7_file_base+"v10_50_demos/run_"+num2str(i-1)+"/results.csv");

    one_demos_mean{i} = mean(one_demos_raw{i},2)/episode_length;
    two_demos_mean{i} = mean(two_demos_raw{i},2)/episode_length;
    five_demos_mean{i} = mean(five_demos_raw{i},2)/episode_length;
    ten_demos_mean{i} = mean(ten_demos_raw{i},2)/episode_length;
    twenty_demos_mean{i} = mean(twenty_demos_raw{i},2)/episode_length;
    fifty_demos_mean{i} = mean(fifty_demos_raw{i},2)/episode_length;
end

one_demos_trans = [transpose(one_demos_mean{1}); transpose(one_demos_mean{2}); transpose(one_demos_mean{3}); transpose(one_demos_mean{4}); transpose(one_demos_mean{5})];
two_demos_trans = [transpose(two_demos_mean{1}); transpose(two_demos_mean{2}); transpose(two_demos_mean{3}); transpose(two_demos_mean{4}); transpose(two_demos_mean{5})];
five_demos_trans = [transpose(five_demos_mean{1}); transpose(five_demos_mean{2}); transpose(five_demos_mean{3}); transpose(five_demos_mean{4}); transpose(five_demos_mean{5})];
ten_demos_trans = [transpose(ten_demos_mean{1}); transpose(ten_demos_mean{2}); transpose(ten_demos_mean{3}); transpose(ten_demos_mean{4}); transpose(ten_demos_mean{5})];
twenty_demos_trans = [transpose(twenty_demos_mean{1}); transpose(twenty_demos_mean{2}); transpose(twenty_demos_mean{3}); transpose(twenty_demos_mean{4}); transpose(twenty_demos_mean{5})];
fifty_demos_trans = [transpose(fifty_demos_mean{1}); transpose(fifty_demos_mean{2}); transpose(fifty_demos_mean{3}); transpose(fifty_demos_mean{4}); transpose(fifty_demos_mean{5})];


one_demos_overall_mean = mean(mean(one_demos_trans));
two_demos_overall_mean = mean(mean(two_demos_trans));
five_demos_overall_mean = mean(mean(five_demos_trans));
ten_demos_overall_mean = mean(mean(ten_demos_trans));
twenty_demos_overall_mean = mean(mean(twenty_demos_trans));
fifty_demos_overall_mean = mean(mean(fifty_demos_trans));

one_demos_overall_var = mean(var(one_demos_trans));
two_demos_overall_var = mean(var(two_demos_trans));
five_demos_overall_var = mean(var(five_demos_trans));
ten_demos_overall_var = mean(var(ten_demos_trans));
twenty_demos_overall_var = mean(var(twenty_demos_trans));
fifty_demos_overall_var = mean(var(fifty_demos_trans));

bar_data = [five_demos_overall_mean ten_demos_overall_mean twenty_demos_overall_mean fifty_demos_overall_mean];
error_data = [five_demos_overall_var ten_demos_overall_var twenty_demos_overall_var fifty_demos_overall_var];

td7_x_values = (0:evaluation_frequency*episode_length/1000:maximum_timesteps/1000);
alpha  = 0.3;
line_width = 8;
error = 'var';

options_2.color_area = [0.8500 0.3250 0.0980];
options_2.color_line = [0.8500 0.3250 0.0980];
options_2.alpha      = alpha;
options_2.line_width = line_width;
options_2.error      = error;
options_2.x_axis     = td7_x_values;

options_3.color_area = [0.9290 0.6940 0.1250];
options_3.color_line = [0.9290 0.6940 0.1250];
options_3.alpha      = alpha;
options_3.line_width = line_width;
options_3.error      = error;
options_3.x_axis     = td7_x_values;

options_4.color_area = [0.4660 0.6740 0.1880];
options_4.color_line = [0.4660 0.6740 0.1880];
options_4.alpha      = alpha;
options_4.line_width = line_width;
options_4.error      = error;
options_4.x_axis     = td7_x_values;

options_5.color_area = [0.4940 0.1840 0.5560];
options_5.color_line = [0.4940 0.1840 0.5560];
options_5.alpha      = alpha;
options_5.line_width = line_width;
options_5.error      = error;
options_5.x_axis     = td7_x_values;

options_6.color_area = [21 5 120]./255;
options_6.color_line = [21 5 120]./255;
options_6.alpha      = alpha;
options_6.line_width = line_width;
options_6.error      = error;
options_6.x_axis     = td7_x_values;

options_7.color_area = [0 255 255]./255;
options_7.color_line = [0 255 255]./255;
options_7.alpha      = alpha;
options_7.line_width = line_width;
options_7.error      = error;
options_7.x_axis     = td7_x_values;


figure('Position',[0 0 1920 1440]);
hold on;
% bar(bar_data)
% set(gca,'xticklabel',{'10%','20%','50%','75%'});

X = categorical({'5','10', '20', '50'});
X = reordercats(X,{'5','10', '20', '50'});
% Y = [10 21 33 52];
b = bar(X,bar_data);
% b.LineWidth = line_width;
b.FaceAlpha = 0.9;
b.FaceColor = 'flat';
% b.CData(1,:) = options_2.color_line;
% b.CData(2,:) = options_6.color_line;
b.CData(1,:) = options_2.color_line;
b.CData(2,:) = options_3.color_line;
b.CData(3,:) = options_4.color_line;
b.CData(4,:) = options_5.color_line;

e = errorbar(X, bar_data, error_data);
e.Color = 'black';
e.LineStyle = 'none';
e.LineWidth = line_width;
e.CapSize = 40;

%xlabel('Episodes in Demonstration Buffer')
ylabel('Mean Reward and Variance','FontSize',16)
ylim([0.4 0.7])

% set(gca,'FontSize',18)
set(gca,'FontSize',55)
hold off;


% td7_x_values = (0:evaluation_frequency*episode_length/1000:maximum_timesteps/1000);
% alpha  = 0.3;
% line_width = 4;
% error = 'var';




figure('Position',[0 0 1920 1440]);
hold on
plot_areaerrorbar(five_demos_trans, options_2);
plot_areaerrorbar(ten_demos_trans, options_3);
plot_areaerrorbar(twenty_demos_trans, options_4);
plot_areaerrorbar(fifty_demos_trans, options_5);
%plot_areaerrorbar(one_demos_trans, options_2);
%plot_areaerrorbar(two_demos_trans, options_6);


% xlim([0 26])
ylim([0 1])
%lgd = legend('', '10 Demos', '', '20 Demos', '', '50 Demos', '', '5 Demos', 'Location','best');
%lgd.NumColumns = 3;
xlabel('Time Steps (1K)','FontSize',16)
ylabel('Normalized Reward','FontSize',16)
set(gca,'FontSize',55)
%set(gca,'FontSize',18)

%title('Variable Impedance Controller Comparison')
hold off

%% Comparing influence of demonstration to exploration data ratio for learning

clear all;
close all;

td7_file_base = "door_mirror/gh360/osc_pose/online/";
runs = 5;
episode_length = 500;
evaluation_frequency = 10;
maximum_timesteps = episode_length * 200;

ten_percenatage_raw = cell(1, runs);
ten_percenatage_mean = cell(1,runs);
twentyfive_percenatage_raw = cell(1, runs);
twentyfive_percenatage_mean = cell(1,runs);
fifty_percenatage_raw = cell(1, runs);
fifty_percenatage_mean = cell(1,runs);
seventyfive_percenatage_raw = cell(1, runs);
seventyfive_percenatage_mean = cell(1,runs);

% twenty_demos_raw = cell(1, runs);
% twenty_demos_mean = cell(1,runs);
% fifty_demos_raw = cell(1, runs);
% fifty_demos_mean = cell(1,runs);

for i=1:runs
    ten_percenatage_raw{i} = readmatrix(td7_file_base+"v13_10_ratio/run_"+num2str(i-1)+"/results.csv");
    twentyfive_percenatage_raw{i} = readmatrix(td7_file_base+"v11_25_ratio/run_"+num2str(i-1)+"/results.csv");
    fifty_percenatage_raw{i} = readmatrix(td7_file_base+"v9_20_demos/run_"+num2str(i-1)+"/results.csv");
    seventyfive_percenatage_raw{i} = readmatrix(td7_file_base+"v12_75_ratio/run_"+num2str(i-1)+"/results.csv");

    ten_percenatage_mean{i} = mean(ten_percenatage_raw{i},2)/episode_length;
    twentyfive_percenatage_mean{i} = mean(twentyfive_percenatage_raw{i},2)/episode_length;
    fifty_percenatage_mean{i} = mean(fifty_percenatage_raw{i},2)/episode_length;
    seventyfive_percenatage_mean{i} = mean(seventyfive_percenatage_raw{i},2)/episode_length;
end

% ten_percenatage_overall_mean = mean(ten_percenatage_mean);

ten_percenatage_trans = [transpose(ten_percenatage_mean{1}); transpose(ten_percenatage_mean{2}); transpose(ten_percenatage_mean{3}); transpose(ten_percenatage_mean{4}); transpose(ten_percenatage_mean{5})];
twentyfive_percenatage_trans = [transpose(twentyfive_percenatage_mean{1}); transpose(twentyfive_percenatage_mean{2}); transpose(twentyfive_percenatage_mean{3}); transpose(twentyfive_percenatage_mean{4}); transpose(twentyfive_percenatage_mean{5})];
fifty_percenatage_trans = [transpose(fifty_percenatage_mean{1}); transpose(fifty_percenatage_mean{2}); transpose(fifty_percenatage_mean{3}); transpose(fifty_percenatage_mean{4}); transpose(fifty_percenatage_mean{5})];
seventyfive_percenatage_trans = [transpose(seventyfive_percenatage_mean{1}); transpose(seventyfive_percenatage_mean{2}); transpose(seventyfive_percenatage_mean{3}); transpose(seventyfive_percenatage_mean{4}); transpose(seventyfive_percenatage_mean{5})];

ten_percenatage_overall_mean = mean(mean(ten_percenatage_trans));
twentyfive_percenatage_overall_mean = mean(mean(twentyfive_percenatage_trans));
fifty_percenatage_overall_mean = mean(mean(fifty_percenatage_trans));
seventyfive_percenatage_overall_mean = mean(mean(seventyfive_percenatage_trans));

ten_percenatage_overall_var = mean(var(ten_percenatage_trans));
twentyfive_percenatage_overall_var = mean(var(twentyfive_percenatage_trans));
fifty_percenatage_overall_var = mean(var(fifty_percenatage_trans));
seventyfive_percenatage_overall_var = mean(var(seventyfive_percenatage_trans));

bar_data = [ten_percenatage_overall_mean twentyfive_percenatage_overall_mean fifty_percenatage_overall_mean seventyfive_percenatage_overall_mean];
error_data = [ten_percenatage_overall_var twentyfive_percenatage_overall_var fifty_percenatage_overall_var seventyfive_percenatage_overall_var];

td7_x_values = (0:evaluation_frequency*episode_length/1000:maximum_timesteps/1000);
alpha  = 0.3;
line_width = 8;
error = 'var';

options_2.color_area = [0.8500 0.3250 0.0980];
options_2.color_line = [0.8500 0.3250 0.0980];
options_2.alpha      = alpha;
options_2.line_width = line_width;
options_2.error      = error;
options_2.x_axis     = td7_x_values;

options_3.color_area = [0.9290 0.6940 0.1250];
options_3.color_line = [0.9290 0.6940 0.1250];
options_3.alpha      = alpha;
options_3.line_width = line_width;
options_3.error      = error;
options_3.x_axis     = td7_x_values;

options_4.color_area = [0.4660 0.6740 0.1880];
options_4.color_line = [0.4660 0.6740 0.1880];
options_4.alpha      = alpha;
options_4.line_width = line_width;
options_4.error      = error;
options_4.x_axis     = td7_x_values;

options_5.color_area = [0.4940 0.1840 0.5560];
options_5.color_line = [0.4940 0.1840 0.5560];
options_5.alpha      = alpha;
options_5.line_width = line_width;
options_5.error      = error;
options_5.x_axis     = td7_x_values;


figure('Position',[0 0 1920 1440]);
hold on;
% bar(bar_data)
% set(gca,'xticklabel',{'10%','20%','50%','75%'});

X = categorical({'10%','20%','50%','75%'});
X = reordercats(X,{'10%','20%','50%','75%'});
% Y = [10 21 33 52];
b = bar(X,bar_data);
% b.LineWidth = line_width;
b.FaceAlpha = 0.9;
b.FaceColor = 'flat';
b.CData(1,:) = options_2.color_line;
b.CData(2,:) = options_3.color_line;
b.CData(3,:) = options_4.color_line;
b.CData(4,:) = options_5.color_line;

e = errorbar(X, bar_data, error_data);
e.Color = 'black';
e.LineStyle = 'none';
e.LineWidth = line_width;
e.CapSize = 40;

title('Percentage of Demonstration Data in Learning Batch')
%xlabel('Demonstrations in Learning Batch')
ylabel('Mean Reward and Variance','FontSize',16)
ylim([0.5 0.7])

% set(gca,'FontSize',18)
set(gca,'FontSize',55)
hold off;



% td7_x_values = (0:evaluation_frequency*episode_length/1000:maximum_timesteps/1000);
% alpha  = 0.3;
% line_width = 4;
% error = 'var';




figure('Position',[0 0 1920 1440]);
hold on
plot_areaerrorbar(ten_percenatage_trans, options_2);
plot_areaerrorbar(twentyfive_percenatage_trans, options_3);
plot_areaerrorbar(fifty_percenatage_trans, options_4);
plot_areaerrorbar(seventyfive_percenatage_trans, options_5);

% xlim([0 26])
ylim([0 1])
% lgd = legend('', '10% ratio', '', '20% ratio', '', '50% ratio', '', '75% ratio', 'Location','best');
%lgd.NumColumns = 3;
xlabel('Time Steps (1K)','FontSize',16)
ylabel('Normalized Reward','FontSize',16)
set(gca,'FontSize',55)
% set(gca,'FontSize',18)

%title('Variable Impedance Controller Comparison')
hold off


%%
clear all;
close all;

% td7_file_base = "lift/panda/osc_pose/online/";
% td7_file_base = "door/real_gh360/eef_vel/online/";
td7_file_base = "door_mirror/gh360/osc_pose/online/";
% rl_with_demo_run_0_raw = readmatrix(td7_file_base+"v5/run_0/results.csv");
% rl_with_demo_run_1_raw = readmatrix(td7_file_base+"v13_corl_with_demos_4/run_1/results.csv");
% rl_with_demo_run_2_raw = readmatrix(td7_file_base+"v12_corl_with_demos_3/run_2/results.csv");
% rl_with_demo_run_3_raw = readmatrix(td7_file_base+"v12_corl_with_demos_3/run_3/results.csv");
% rl_with_demo_run_4_raw = readmatrix(td7_file_base+"v12_corl_with_demos_3/run_4/results.csv");

rl_without_demo_no_variance_run_0_raw = readmatrix(td7_file_base+"v4_rl_no_variance/run_0/results.csv");
rl_without_demo_no_variance_run_1_raw = readmatrix(td7_file_base+"v4_rl_no_variance/run_1/results.csv");
rl_without_demo_no_variance_run_2_raw = readmatrix(td7_file_base+"v4_rl_no_variance/run_2/results.csv");
rl_without_demo_no_variance_run_3_raw = readmatrix(td7_file_base+"v4_rl_no_variance/run_3/results.csv");
rl_without_demo_no_variance_run_4_raw = readmatrix(td7_file_base+"v4_rl_no_variance/run_4/results.csv");

rl_without_demo_with_variance_run_0_raw = readmatrix(td7_file_base+"v6_rl_with_variance/run_0/results.csv");
rl_without_demo_with_variance_run_1_raw = readmatrix(td7_file_base+"v6_rl_with_variance/run_1/results.csv");
rl_without_demo_with_variance_run_2_raw = readmatrix(td7_file_base+"v6_rl_with_variance/run_2/results.csv");
rl_without_demo_with_variance_run_3_raw = readmatrix(td7_file_base+"v6_rl_with_variance/run_3/results.csv");
rl_without_demo_with_variance_run_4_raw = readmatrix(td7_file_base+"v6_rl_with_variance/run_4/results.csv");

rl_with_demo_no_variance_run_0_raw = readmatrix(td7_file_base+"v5_rl_with_demo_no_variance/run_0/results.csv");
rl_with_demo_no_variance_run_1_raw = readmatrix(td7_file_base+"v5_rl_with_demo_no_variance/run_1/results.csv");
rl_with_demo_no_variance_run_2_raw = readmatrix(td7_file_base+"v5_rl_with_demo_no_variance/run_2/results.csv");
rl_with_demo_no_variance_run_3_raw = readmatrix(td7_file_base+"v5_rl_with_demo_no_variance/run_3/results.csv");
rl_with_demo_no_variance_run_4_raw = readmatrix(td7_file_base+"v5_rl_with_demo_no_variance/run_4/results.csv");

rl_with_demo_with_variance_run_0_raw = readmatrix(td7_file_base+"v7_rl_with_demo_with_variance/run_0/results.csv");
rl_with_demo_with_variance_run_1_raw = readmatrix(td7_file_base+"v7_rl_with_demo_with_variance/run_1/results.csv");
rl_with_demo_with_variance_run_2_raw = readmatrix(td7_file_base+"v7_rl_with_demo_with_variance/run_2/results.csv");
rl_with_demo_with_variance_run_3_raw = readmatrix(td7_file_base+"v7_rl_with_demo_with_variance/run_3/results.csv");
rl_with_demo_with_variance_run_4_raw = readmatrix(td7_file_base+"v7_rl_with_demo_with_variance/run_4/results.csv");


episode_length = 500;
evaluation_frequency = 10;
maximum_timesteps = episode_length * 200;


rl_without_demo_no_variance_run_0_mean = mean(rl_without_demo_no_variance_run_0_raw,2);
rl_without_demo_no_variance_run_1_mean = mean(rl_without_demo_no_variance_run_1_raw,2);
rl_without_demo_no_variance_run_2_mean = mean(rl_without_demo_no_variance_run_2_raw,2);
rl_without_demo_no_variance_run_3_mean = mean(rl_without_demo_no_variance_run_3_raw,2);
rl_without_demo_no_variance_run_4_mean = mean(rl_without_demo_no_variance_run_4_raw,2);

rl_without_demo_with_variance_run_0_mean = mean(rl_without_demo_with_variance_run_0_raw,2);
rl_without_demo_with_variance_run_1_mean = mean(rl_without_demo_with_variance_run_1_raw,2);
rl_without_demo_with_variance_run_2_mean = mean(rl_without_demo_with_variance_run_2_raw,2);
rl_without_demo_with_variance_run_3_mean = mean(rl_without_demo_with_variance_run_3_raw,2);
rl_without_demo_with_variance_run_4_mean = mean(rl_without_demo_with_variance_run_4_raw,2);

rl_with_demo_no_variance_run_0_mean = mean(rl_with_demo_no_variance_run_0_raw,2);
rl_with_demo_no_variance_run_1_mean = mean(rl_with_demo_no_variance_run_1_raw,2);
rl_with_demo_no_variance_run_2_mean = mean(rl_with_demo_no_variance_run_2_raw,2);
rl_with_demo_no_variance_run_3_mean = mean(rl_with_demo_no_variance_run_3_raw,2);
rl_with_demo_no_variance_run_4_mean = mean(rl_with_demo_no_variance_run_4_raw,2);

rl_with_demo_with_variance_run_0_mean = mean(rl_with_demo_with_variance_run_0_raw,2);
rl_with_demo_with_variance_run_1_mean = mean(rl_with_demo_with_variance_run_1_raw,2);
rl_with_demo_with_variance_run_2_mean = mean(rl_with_demo_with_variance_run_2_raw,2);
rl_with_demo_with_variance_run_3_mean = mean(rl_with_demo_with_variance_run_3_raw,2);
rl_with_demo_with_variance_run_4_mean = mean(rl_with_demo_with_variance_run_4_raw,2);

rl_without_demo_no_variance_run_0_mean = rl_without_demo_no_variance_run_0_mean/episode_length;
rl_without_demo_no_variance_run_1_mean = rl_without_demo_no_variance_run_1_mean/episode_length;
rl_without_demo_no_variance_run_2_mean = rl_without_demo_no_variance_run_2_mean/episode_length;
rl_without_demo_no_variance_run_3_mean = rl_without_demo_no_variance_run_3_mean/episode_length;
rl_without_demo_no_variance_run_4_mean = rl_without_demo_no_variance_run_4_mean/episode_length;

rl_without_demo_with_variance_run_0_mean = rl_without_demo_with_variance_run_0_mean/episode_length;
rl_without_demo_with_variance_run_1_mean = rl_without_demo_with_variance_run_1_mean/episode_length;
rl_without_demo_with_variance_run_2_mean = rl_without_demo_with_variance_run_2_mean/episode_length;
rl_without_demo_with_variance_run_3_mean = rl_without_demo_with_variance_run_3_mean/episode_length;
rl_without_demo_with_variance_run_4_mean = rl_without_demo_with_variance_run_4_mean/episode_length;

rl_with_demo_no_variance_run_0_mean = rl_with_demo_no_variance_run_0_mean/episode_length;
rl_with_demo_no_variance_run_1_mean = rl_with_demo_no_variance_run_1_mean/episode_length;
rl_with_demo_no_variance_run_2_mean = rl_with_demo_no_variance_run_2_mean/episode_length;
rl_with_demo_no_variance_run_3_mean = rl_with_demo_no_variance_run_3_mean/episode_length;
rl_with_demo_no_variance_run_4_mean = rl_with_demo_no_variance_run_4_mean/episode_length;

rl_with_demo_with_variance_run_0_mean = rl_with_demo_with_variance_run_0_mean/episode_length;
rl_with_demo_with_variance_run_1_mean = rl_with_demo_with_variance_run_1_mean/episode_length;
rl_with_demo_with_variance_run_2_mean = rl_with_demo_with_variance_run_2_mean/episode_length;
rl_with_demo_with_variance_run_3_mean = rl_with_demo_with_variance_run_3_mean/episode_length;
rl_with_demo_with_variance_run_4_mean = rl_with_demo_with_variance_run_4_mean/episode_length;

td7_x_values = (0:evaluation_frequency*episode_length/1000:maximum_timesteps/1000);
offset = episode_length*20/1000;
td7_x_values2 = (offset:evaluation_frequency*episode_length/1000:maximum_timesteps/1000+offset);

alpha  = 0.3;
line_width = 4;
error = 'var';


rl_without_demo_no_variance_trans = [transpose(rl_without_demo_no_variance_run_0_mean); transpose(rl_without_demo_no_variance_run_1_mean); transpose(rl_without_demo_no_variance_run_2_mean); transpose(rl_without_demo_no_variance_run_3_mean); transpose(rl_without_demo_no_variance_run_4_mean)];
rl_without_demo_with_variance_trans = [transpose(rl_without_demo_with_variance_run_0_mean); transpose(rl_without_demo_with_variance_run_1_mean); transpose(rl_without_demo_with_variance_run_2_mean); transpose(rl_without_demo_with_variance_run_3_mean); transpose(rl_without_demo_with_variance_run_4_mean)];
rl_with_demo_no_variance_trans = [transpose(rl_with_demo_no_variance_run_0_mean); transpose(rl_with_demo_no_variance_run_1_mean); transpose(rl_with_demo_no_variance_run_2_mean); transpose(rl_with_demo_no_variance_run_3_mean); transpose(rl_with_demo_no_variance_run_4_mean)];
rl_with_demo_with_variance_trans = [transpose(rl_with_demo_with_variance_run_0_mean); transpose(rl_with_demo_with_variance_run_1_mean); transpose(rl_with_demo_with_variance_run_2_mean); transpose(rl_with_demo_with_variance_run_3_mean); transpose(rl_with_demo_with_variance_run_4_mean)];

% exp_2_trans = [transpose(exp_2_mean)];
% exp_3_trans = [transpose(exp_3_mean)];
% exp_4_trans = [transpose(exp_4_mean)];
% exp_5_trans = [transpose(exp_5_mean)];

% td7_reward_shaping_online = [transpose(td7_reward_shaping_online_run_0_mean);transpose(td7_reward_shaping_online_run_1_mean);transpose(td7_reward_shaping_online_run_2_mean)];
% sac_reward_shaping_no_demo = [transpose(sac_reward_shaping_no_demo_run_0_mean);transpose(sac_reward_shaping_no_demo_run_1_mean);transpose(sac_reward_shaping_no_demo_run_2_mean)];

%sac_reward_shaping_with_demo = [transpose(sac_reward_shaping_with_demo_run_0_mean); transpose(sac_reward_shaping_with_demo_run_1_mean); transpose(sac_reward_shaping_with_demo_run_2_mean)];
%sac_reward_shaping_pre_supervised = [transpose(sac_reward_shaping_pre_supervised_run_0_mean);transpose(sac_reward_shaping_pre_supervised_run_1_mean);transpose(sac_reward_shaping_pre_supervised_run_2_mean)];
%sac_reward_shaping_pre_supervised_less_std = [transpose(sac_reward_shaping_pre_supervised_less_std_run_0_mean);transpose(sac_reward_shaping_pre_supervised_less_std_run_1_mean);transpose(sac_reward_shaping_pre_supervised_less_std_run_2_mean)];
%sac_no_reward_shaping_no_demo = [transpose(sac_no_reward_shaping_no_demo_run_0_mean);transpose(sac_no_reward_shaping_no_demo_run_1_mean);transpose(sac_no_reward_shaping_no_demo_run_2_mean)];


options_2.color_area = [0.8500 0.3250 0.0980];
options_2.color_line = [0.8500 0.3250 0.0980];
options_2.alpha      = alpha;
options_2.line_width = line_width;
options_2.error      = error;
options_2.x_axis     = td7_x_values;

options_3.color_area = [0.9290 0.6940 0.1250];
options_3.color_line = [0.9290 0.6940 0.1250];
options_3.alpha      = alpha;
options_3.line_width = line_width;
options_3.error      = error;
options_3.x_axis     = td7_x_values;

options_4.color_area = [0.4660 0.6740 0.1880];
options_4.color_line = [0.4660 0.6740 0.1880];
options_4.alpha      = alpha;
options_4.line_width = line_width;
options_4.error      = error;
options_4.x_axis     = td7_x_values;

options_5.color_area = [0.4940 0.1840 0.5560];
options_5.color_line = [0.4940 0.1840 0.5560];
options_5.alpha      = alpha;
options_5.line_width = line_width;
options_5.error      = error;
options_5.x_axis     = td7_x_values;

figure('Position',[0 0 1920 1440]);
hold on
% plot_areaerrorbar(exp_2_trans, options_3);
plot_areaerrorbar(rl_without_demo_no_variance_trans, options_5);
plot_areaerrorbar(rl_without_demo_with_variance_trans, options_3);
plot_areaerrorbar(rl_with_demo_no_variance_trans, options_4);
plot_areaerrorbar(rl_with_demo_with_variance_trans, options_2);


%plot_areaerrorbar(exp_3_trans, options_4);
%plot_areaerrorbar(exp_1_trans, options_2);




% xlim([0 26])
ylim([0 1])
lgd = legend('', 'No Demo; No Variance', '', 'No Demo; With Variance', '', 'With Demo; No Variance', '', 'With Demo; With Variance', 'Location','best');
%lgd.NumColumns = 3;
xlabel('Time Steps (1K)','FontSize',16)
ylabel('Normalized Reward','FontSize',16)
%set(gca,'FontSize',55)
set(gca,'FontSize',18)

%title('Variable Impedance Controller Comparison')
hold off

%% Comparing full length learning
clear all;
close all;

td7_file_base = "door_mirror/gh360/osc_pose/";
runs = 5;
episode_length = 500;
evaluation_frequency = 10;
episodes = 10000;
mean_plot = false;



file_paths = [
    td7_file_base+"/online/"+"v17_rl_with_demo_full_length";
    td7_file_base+"/online/"+"v18_rl_without_demo_full_length";
    td7_file_base+"/offline/"+"v4_medium_expert_paths_full_length";
];

bar_categories = {''};
bar_category_label = '';

data = draw_learning_analysis(file_paths, runs, episode_length, evaluation_frequency, episodes, mean_plot, bar_categories, bar_category_label);

%% Comparing different learning methods
clear all;
close all;

td7_file_base = "door_mirror/gh360/osc_pose/";
runs = 5;
episode_length = 500;
evaluation_frequency = 10;
episodes = 200;
mean_plot = false;

file_paths = [
    td7_file_base+"/online/"+"v9_20_demos"; % Online RL + Demo
    td7_file_base+"/online/"+"v19_rl_without_demo"; % Online RL
    td7_file_base+"/offline/"+"v3_medium_expert_random_paths"; % Offline
    % BC
];

bar_categories = {''};
bar_category_label = '';

data = draw_learning_analysis(file_paths, runs, episode_length, evaluation_frequency, episodes, mean_plot, bar_categories, bar_category_label);


%% Comparing Demonstration to Exploration ratio in learning batch
clear all;
close all;

td7_file_base = "door_mirror/gh360/osc_pose/online/";
runs = 5;
episode_length = 500;
evaluation_frequency = 10;
episodes = 200;
mean_plot = true;



file_paths = [
    td7_file_base+"v13_10_ratio";
    td7_file_base+"v11_25_ratio";
    td7_file_base+"v9_20_demos";
    td7_file_base+"v12_75_ratio";
];

bar_categories = {'10%', '25%', '50%', '75%'};
bar_category_label = 'Demonstrations in Learning Batch';

data = draw_learning_analysis(file_paths, runs, episode_length, evaluation_frequency, episodes, mean_plot, bar_categories, bar_category_label);

%%
function option = getOption(iter, episode_length, evaluation_frequency, maximum_timesteps)
    td7_x_values = (0:evaluation_frequency*episode_length/1000:maximum_timesteps/1000);
    alpha  = 0.3;
    line_width = 8;
    error = 'std';
    colors = {[0.8500 0.3250 0.0980]; [0.9290 0.6940 0.1250]; [0.4660 0.6740 0.1880]; [0.4940 0.1840 0.5560]; [21 5 120]./255; [0 255 255]./255};

    option.color_area = colors{iter};
    option.color_line = colors{iter};
    option.alpha = alpha;
    option.line_width = line_width;
    option.error      = error;
    option.x_axis     = td7_x_values;
   
end

function data_mean = draw_learning_analysis(file_paths, runs, episode_length, evaluation_frequency, episodes, mean_plot, bar_categories, bar_category_label)
    maximum_timesteps = episode_length * episodes;
    evaluations = episodes/evaluation_frequency+1;

    data = cell(size(file_paths,1), runs);
    data_mean = cell(size(file_paths,1), runs);
    data_trans = cell(size(file_paths,1),1);
    
    demo = [303.58425876378936 379.84050961932354 393.9671332686742 305.3618362973439 312.18631963501014 335.21817626120776 321.10422960815595 322.1877051093933 387.4116885320301 373.05563995501586 319.3999190978943 268.59024276014486 369.8436111152886 355.0600473635285 367.17086653023966 297.33909141111695 346.05007938467986 331.2123322407663 345.772363427121 366.36275313640635 351.64756210367517 270.44162276974953 396.23024372545103 320.5808254673216 357.7309294815964 360.9186174750906 325.50420508330194 370.0506123849263 361.14031763162535 320.02624306602286 322.20844666327184 319.7311495338364 382.79713636410816 289.5285159767229 346.9746182927988 368.9169901236438 321.72697232517726 327.2626139221626 350.3050501240054 377.3064498479384 321.12500925035374 316.6972973286751 328.22238841747077 353.1643804368879 365.9145708538034 327.42012712492135 316.7627085880502 359.9344037359584 350.9696326816663 291.3304514085871];
    demo = demo/episode_length;
%     demo = demo(1:20)/episode_length;
    size(demo,2)
    demo_trans = zeros(size(demo,2), episodes/10+1);
    size(demo_trans)
    for i=1:(episodes/10+1)
        demo_trans(:,i) = transpose(demo);
    end
%     demo_trans = [transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo)];
%     demo_trans = [transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo) transpose(demo)];


    figure('Position',[0 0 1920 1440]);
    hold on
    for i=1:size(file_paths)
        data_trans{i} = zeros(runs,evaluations);
        for j=1:runs
            data{i,j} = readmatrix(file_paths(i)+"/run_"+num2str(j-1)+"/results.csv");
            data_mean{i,j} = mean(data{i,j},2)/episode_length;
            data_trans{i}(j,:) = transpose(data_mean{i,j});
        end
        
        plot_areaerrorbar(data_trans{i}, getOption(i, episode_length, evaluation_frequency, maximum_timesteps));
    end

    plot_areaerrorbar(demo_trans, getOption(size(file_paths)+1, episode_length, evaluation_frequency, maximum_timesteps));

    %xlim([0 300])
    ylim([0 1])
    % lgd = legend('', '10% ratio', '', '20% ratio', '', '50% ratio', '', '75% ratio', 'Location','best');
    %lgd.NumColumns = 3;
    xlabel('Time Steps (1K)','FontSize',16)
    ylabel('Normalized Reward','FontSize',16)
    set(gca,'FontSize',55)
    % set(gca,'FontSize',18)
    
    %title('Variable Impedance Controller Comparison')
    hold off

    if mean_plot == true
        line_width = 8;
        data_overall_mean = zeros(size(file_paths,1),1);
        data_overall_mean_var = zeros(size(file_paths,1),1);

        figure('Position',[0 0 1920 1440]);
        hold on; 

        for i=1:size(file_paths)
            data_overall_mean(i) = mean(mean(data_trans{i}));
            data_overall_mean_var(i) = mean(var(data_trans{i}));
        end

        X = categorical(bar_categories);
        X = reordercats(X,bar_categories);
        % Y = [10 21 33 52];
        b = bar(X,data_overall_mean);
        % b.LineWidth = line_width;
        b.FaceAlpha = 0.9;
        b.FaceColor = 'flat';

        for i=1:size(file_paths)
            b.CData(i,:) = getOption(i, episode_length, evaluation_frequency, maximum_timesteps).color_line;
        end

        e = errorbar(X, data_overall_mean, data_overall_mean_var);
        e.Color = 'black';
        e.LineStyle = 'none';
        e.LineWidth = line_width;
        e.CapSize = 40;

        xlabel(bar_category_label)
        ylabel('Mean Reward and Variance','FontSize',16)
        ylim([0.5 0.7])
        
        % set(gca,'FontSize',18)
        set(gca,'FontSize',55)
        hold off;
    end

end



function avg = calcAverage(x)
    n = 20;             % Number of elements to create the mean over
    s1 = size(x, 1);      % Find the next smaller multiple of n
    m  = s1 - mod(s1, n);
    y  = reshape(x(1:m), n, []);     % Reshape x to a [n, m/n] matrix
    avg = transpose(sum(y, 1) / n);  % Calculate the mean over the 1st dim
end


% ----------------------------------------------------------------------- %
% Function plot_areaerrorbar plots the mean and standard deviation of a   %
% set of data filling the space between the positive and negative mean    %
% error using a semi-transparent background, completely customizable.     %
%                                                                         %
%   Input parameters:                                                     %
%       - data:     Data matrix, with rows corresponding to observations  %
%                   and columns to samples.                               %
%       - options:  (Optional) Struct that contains the customized params.%
%           * options.handle:       Figure handle to plot the result.     %
%           * options.color_area:   RGB color of the filled area.         %
%           * options.color_line:   RGB color of the mean line.           %
%           * options.alpha:        Alpha value for transparency.         %
%           * options.line_width:   Mean line width.                      %
%           * options.x_axis:       X time vector.                        %
%           * options.error:        Type of error to plot (+/-).          %
%                   if 'std',       one standard deviation;               %
%                   if 'sem',       standard error mean;                  %
%                   if 'var',       one variance;                         %
%                   if 'c95',       95% confidence interval.              %
% ----------------------------------------------------------------------- %
%   Example of use:                                                       %
%       data = repmat(sin(1:0.01:2*pi),100,1);                            %
%       data = data + randn(size(data));                                  %
%       plot_areaerrorbar(data);                                          %
% ----------------------------------------------------------------------- %
%   Author:  Victor Martinez-Cagigal                                      %
%   Date:    30/04/2018                                                   %
%   E-mail:  vicmarcag (at) gmail (dot) com                               %
% ----------------------------------------------------------------------- %
function plot_areaerrorbar(data, options)
    % Default options
    if(nargin<2)
        options.handle     = figure(1);
        options.color_area = [128 193 219]./255;    % Blue theme
        options.color_line = [ 52 148 186]./255;
        %options.color_area = [243 169 114]./255;    % Orange theme
        %options.color_line = [236 112  22]./255;
        options.alpha      = 0.5;
        options.line_width = 2;
        options.error      = 'std';
    end
    if(isfield(options,'x_axis')==0), options.x_axis = 1:size(data,2); end
    options.x_axis = options.x_axis(:);
    
    % Computing the mean and standard deviation of the data matrix
    data_mean = mean(data,1);
    data_std  = std(data,0,1);
    
    % Type of error plot
    switch(options.error)
        case 'std', error = data_std;
        case 'sem', error = (data_std./sqrt(size(data,1)));
        case 'var', error = (data_std.^2);
        case 'c95', error = (data_std./sqrt(size(data,1))).*1.96;
    end
    
    % Plotting the result
    %figure(options.handle);
    x_vector = [options.x_axis', fliplr(options.x_axis')];
    patch = fill(x_vector, [data_mean+error,fliplr(data_mean-error)], options.color_area);
    set(patch, 'edgecolor', 'none');
    set(patch, 'FaceAlpha', options.alpha);
    %hold on;
    plot(options.x_axis, data_mean, 'color', options.color_line, ...
        'LineWidth', options.line_width);
    %hold off;
    
end

















