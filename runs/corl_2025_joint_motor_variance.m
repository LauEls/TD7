clear all;
close all;

td7_file_base = "door/real_gh360/eef_vel/online/";
joint_states = readmatrix(td7_file_base+"v8_corl_with_demos/run_2/joint_states.csv");
motor_states = readmatrix(td7_file_base+"v8_corl_with_demos/run_2/motor_states.csv");

%%
close all;

joint_names = ["" "shoulder yaw" "shoulder roll" "shoulder pitch" "upperarm roll" "elbow" "forearm roll" "wrist pitch"];
motor_states_adjusted = zeros(length(motor_states), 8);
motor_states_adjusted(:,1) = motor_states(:,1);
motor_states_adjusted(:,2) = (motor_states(:,2)+motor_states(:,3))/2;
motor_states_adjusted(:,3) = (motor_states(:,4)+motor_states(:,5))/2;
motor_states_adjusted(:,4) = (motor_states(:,6)+motor_states(:,7))/2;
motor_states_adjusted(:,5) = (motor_states(:,8)+motor_states(:,9))/2;
motor_states_adjusted(:,6) = (motor_states(:,10)+motor_states(:,11))/2;
motor_states_adjusted(:,7) = motor_states(:,12);
motor_states_adjusted(:,8) = (motor_states(:,13)+motor_states(:,14))/2;


joint_time_zero = joint_states(1,1);
joint_time_max = joint_states(end,1);
motor_time_zero = motor_states(1,1);
motor_time_max = motor_states(end,1);
if joint_time_zero < motor_time_zero
    time_zero = motor_time_zero;
else
    time_zero = joint_time_zero;
end

if joint_time_max > motor_time_max
    time_max = motor_time_max;
else
    time_max = joint_time_max;
end

%joint_time = (joint_states(:,1)-time_zero)/10^9;
%motor_time = (motor_states(:,1)-time_zero)/10^9;

resample_size = 500000;
t_linear = linspace(time_zero, time_max, resample_size).';
joint_states_resampled = zeros(resample_size, 8);
motor_states_resampled = zeros(resample_size,8);

for i=2:8
    joint_states_resampled(:,i) = interp1(joint_states(:,1), joint_states(:,i), t_linear);
    motor_states_resampled(:,i) = interp1(motor_states(:,1), motor_states_adjusted(:,i), t_linear);

    motor_joint_state = zeros(resample_size, 2);
    motor_joint_state(:,1) = joint_states_resampled(:,i);
    motor_joint_state(:,2) = motor_states_resampled(:,i);

    joint_mean = mean(motor_joint_state);
    joint_var = var(motor_joint_state);
    length1 = length(motor_joint_state(:,1));
    length2 = length(motor_joint_state(:,2));
    X = [motor_joint_state(:,1) motor_joint_state(:,2)];

    %y = mvnpdf(X,joint_mean, joint_var);
    %y = reshape(y, resample_size, resample_size);
    GMModel = fitgmdist(X,1);
    
    % figure;
    % hold on
    % title("Joint: "+joint_names(i))
    % plot(t_linear, motor_states_resampled(:,i));
    % plot(t_linear, joint_states_resampled(:,i));
    % hold off

    figure('Position',[0 0 1920 1440]);
    hold on
    title("Joint: "+joint_names(i));
    xlabel('Joint Angle [rad]','FontSize',16)
    ylabel('Motor Position [rad]','FontSize',16)
    set(gca,'FontSize',55)
    xlim([min(joint_states_resampled(:,i))-0.2 max(joint_states_resampled(:,i))+0.2])
    ylim([min(motor_states_resampled(:,i))-0.2 max(motor_states_resampled(:,i))+0.2])


    gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(GMModel,[x0 y0]),x,y);
    g = gca;
    
    scatter(joint_states_resampled(:,i), motor_states_resampled(:,i), 1,"Marker",".")
    % contour(motor_joint_state(:,1),motor_joint_state(:,2),y,[0.0001 0.001 0.01 0.05 0.15 0.25 0.35]);
    h = fcontour(gmPDF,[g.XLim g.YLim], "LineWidth",3.0);
    plot(joint_mean(1),joint_mean(2), '.', 'Color','red','MarkerSize',20);
    
    hold off

end



%%

resample_rate = 100;
% xypos_resampled = zeros(resample_rate, trialno*2); % make space for multiple trajectories
trialtotest = 1; % say we are interested in trial (trajectory) 1
% Given 1 trajectory data
xypos = [0.0490, 0.0660;
    0.2100, 0.2070;
    0.4500, 0.4700;
    0.6300, 0.8560;
    0.6630, 1.3350;
    0.5440, 1.8850;
    0.3350, 2.4840;
    0.2110, 3.0910;
    0.2150, 3.6720;
    0.3260, 4.2250;
    0.5120, 4.7020;
    0.7960, 4.9150;
    1.1640, 4.8790;
    1.5980, 4.7660;
    2.0980, 4.6230;
    2.6560, 4.4960;
    3.2650, 4.4080;
    3.9080, 4.3560;
    4.5510, 4.3390];
% xypos = cell2mat(tWin_unconcatenated_whole (1, trialtotest)) ;
% xypos_resampled (:,trialtotest*2-1:trialtotest*2) = resample(xypos,resample_rate, length(xypos));
t = (0:size(xypos,1)-1).';
% % [xypos_resampled1,tr1] = resample(xypos, t, 100);
tr2 = linspace(min(t), max(t), 100).';
xypos_resampled2 = interp1(t, xypos, tr2);
figure
plot(xypos(:,1), xypos(:,2), '.-')


%%

mu = [0 0];
Sigma = [0.25 0.3; 0.3 1];

p = mvncdf([0 0],[1 1],mu,Sigma);

