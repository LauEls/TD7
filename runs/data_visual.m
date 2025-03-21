clear all;
close all;

td7_file_base = "lift/panda/osc_pose/online/";
td7_file_base_2 = "door_mirror/gh360/joint_velocity/offline/";

% exp_1_raw = readmatrix(td7_file_base+"v7_reduced_ep_len_50/run_0/results.csv");
% exp_2_raw = readmatrix(td7_file_base+"v7_reduced_ep_len_100/run_0/results.csv");
% exp_3_raw = readmatrix(td7_file_base+"v7_reduced_ep_len_250/run_0/results.csv");
% exp_4_raw = readmatrix(td7_file_base+"v7_reduced_ep_len_250/run_0/results.csv");

exp_1_raw = readmatrix(td7_file_base+"v8_reduced_ep_len_50/run_0/results.csv");
exp_2_raw = readmatrix(td7_file_base+"v8_reduced_ep_len_100/run_0/results.csv");
exp_3_raw = readmatrix(td7_file_base+"v8_reduced_ep_len_250/run_0/results.csv");
exp_4_raw = readmatrix(td7_file_base_2+"v1_first_offline_test//run_0/results.csv");

% exp_1_raw = readmatrix(td7_file_base+"v8_reduced_ep_len_50/run_1/results.csv");
% exp_2_raw = readmatrix(td7_file_base+"v8_reduced_ep_len_100/run_1/results.csv");
% exp_3_raw = readmatrix(td7_file_base+"v8_reduced_ep_len_250/run_1/results.csv");
% exp_4_raw = readmatrix(td7_file_base+"v8_reduced_ep_len_500/run_1/results.csv");

% exp_1_raw = readmatrix(td7_file_base+"v9_demo_buffer_50/run_0/results.csv");
% exp_2_raw = readmatrix(td7_file_base+"v9_demo_buffer_100/run_0/results.csv");
% exp_3_raw = readmatrix(td7_file_base+"v9_demo_buffer_250/run_0/results.csv");
% exp_4_raw = readmatrix(td7_file_base+"v9_demo_buffer_500/run_0/results.csv");

%%
exp_1_mean = mean(exp_1_raw,2);
exp_2_mean = mean(exp_2_raw,2);
exp_3_mean = mean(exp_3_raw,2);
exp_4_mean = mean(exp_4_raw,2);

exp_1_mean = exp_1_mean/50;
exp_2_mean = exp_2_mean/100;
exp_3_mean = exp_3_mean/250;
exp_4_mean = exp_4_mean/500;

% td7_reward_shaping_online_run_0_mean = mean(td7_reward_shaping_online_run_0,2);
% td7_reward_shaping_online_run_1_mean = mean(td7_reward_shaping_online_run_1,2);
% td7_reward_shaping_online_run_2_mean = mean(td7_reward_shaping_online_run_2,2);

td7_x_values = (0:0.005:5);
exp_1_x_values = (0:(50*10/10^6):(50*10000/10^6));
exp_2_x_values = (0:(100*10/10^6):(100*10000/10^6));
exp_3_x_values = (0:(250*10/10^6):(250*10000/10^6));
exp_4_x_values = (0:(500*10/10^6):(500*5000/10^6));
%exp_4_x_values = 0:1:500;

alpha  = 0.3;
line_width = 4;
error = 'std';
%%
close all;

exp_1_trans = [transpose(exp_1_mean)];
exp_2_trans = [transpose(exp_2_mean)];
exp_3_trans = [transpose(exp_3_mean)];
exp_4_trans = [transpose(exp_4_mean)];

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
options_2.x_axis     = exp_1_x_values;

options_3.color_area = [0.9290 0.6940 0.1250];
options_3.color_line = [0.9290 0.6940 0.1250];
options_3.alpha      = alpha;
options_3.line_width = line_width;
options_3.error      = error;
options_3.x_axis     = exp_2_x_values;

options_4.color_area = [0.4660 0.6740 0.1880];
options_4.color_line = [0.4660 0.6740 0.1880];
options_4.alpha      = alpha;
options_4.line_width = line_width;
options_4.error      = error;
options_4.x_axis     = exp_3_x_values;

options_5.color_area = [0.4940 0.1840 0.5560];
options_5.color_line = [0.4940 0.1840 0.5560];
options_5.alpha      = alpha;
options_5.line_width = line_width;
options_5.error      = error;
options_5.x_axis     = exp_4_x_values;

figure('Position',[0 0 1920 1440]);
hold on
plot_areaerrorbar(exp_1_trans, options_2);
plot_areaerrorbar(exp_2_trans, options_3);
plot_areaerrorbar(exp_3_trans, options_4);
plot_areaerrorbar(exp_4_trans, options_5);
xlim([0 5])
ylim([0 1])
lgd = legend('', '50', '', '100', '', '250', '', '500', 'Location','best');
%lgd.NumColumns = 3;
xlabel('Time Steps (1M)','FontSize',16)
ylabel('Total Reward','FontSize',16)
%set(gca,'FontSize',55)
set(gca,'FontSize',18)

%title('Variable Impedance Controller Comparison')
hold off


%%

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

















