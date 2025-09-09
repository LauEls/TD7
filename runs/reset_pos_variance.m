clear all;
close all;

td7_file_base = "door/real_gh360/eef_vel/online/";
joint_states = readmatrix(td7_file_base+"v8_corl_with_demos/run_2/joint_states_test.csv");
motor_states = readmatrix(td7_file_base+"v8_corl_with_demos/run_2/motor_states_test.csv");
goal_eef_vels = readmatrix(td7_file_base+"v8_corl_with_demos/run_2/goal_eef_vel_test.csv");