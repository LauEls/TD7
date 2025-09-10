import rosbag2_py
import numpy as np
import csv
from dataclasses import dataclass
from rclpy.serialization import deserialize_message
from gh360_interfaces.msg import PortStatus
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

@dataclass
class ROSBagMsg:
    time: int
    data: any


# file_base = 'door/real_gh360/eef_vel/online/v8_corl_with_demos/run_2/'
# bag_name = 'rosbag_1745229620'
file_base = 'door/real_gh360/eef_vel/online/v14_video_recording/run_0/'
bag_name = 'rosbag_1746173971'
rosbag_path = file_base + bag_name
joint_states_path = file_base + 'joint_states.csv'
motor_states_path = file_base + 'motor_states.csv'
goal_eef_vel_path = file_base + 'goal_eef_vel.csv'
# joint_states_path = file_base + 'final_eval_joint_states.csv'
# motor_states_path = file_base + 'final_eval_motor_states.csv'
# goal_eef_vel_path = file_base + 'final_eval_goal_eef_vel.csv'


rosbag_reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py._storage.StorageOptions(
    uri=rosbag_path,
    storage_id='sqlite3')
converter_options = rosbag2_py._storage.ConverterOptions('', '')
rosbag_reader.open(storage_options, converter_options)

joint_states = []
motor_states = []
goal_eef_vels = []

while rosbag_reader.has_next():
    topic, msg, t = rosbag_reader.read_next()

    if topic.endswith('/gh360/joint_states'):
        msg_dec = deserialize_message(msg, JointState)
        joint_states.append(ROSBagMsg(t, msg_dec))
    
    elif topic.endswith('/gh360/motor_states_sorted'):
        msg_dec = deserialize_message(msg, PortStatus)
        motor_states.append(ROSBagMsg(t, msg_dec))

    elif topic.endswith('/gh360_control/cmd_eef_vel'):
        msg_dec = deserialize_message(msg, Twist)
        goal_eef_vels.append(ROSBagMsg(t, msg_dec))


print(f"Joint states: {len(joint_states)}")
print(f"Motor states: {len(motor_states)}")
print(f"Goal end-effector velocities: {len(goal_eef_vels)}")

csv_joint_state_positions = np.array([joint_state.data.position for joint_state in joint_states])
csv_joint_state_times = np.array([joint_state.time for joint_state in joint_states])
csv_joint_state_times = csv_joint_state_times.reshape(-1, 1)
csv_joint_states = np.concatenate((csv_joint_state_times, csv_joint_state_positions), axis=1)

csv_motor_states = []
for motor_state in motor_states:
    new_motor_state = []
    new_motor_state.append(motor_state.time)
    for motor in motor_state.data.motors:
        new_motor_state.append(motor.present_position)
    csv_motor_states.append(new_motor_state)
csv_motor_states = np.array(csv_motor_states)

csv_goal_eef_velocities = np.array([[goal_eef_vel.data.linear.x, goal_eef_vel.data.linear.y, goal_eef_vel.data.linear.z, goal_eef_vel.data.angular.x, goal_eef_vel.data.angular.y, goal_eef_vel.data.angular.z] for goal_eef_vel in goal_eef_vels])
csv_goal_eef_vel_times = np.array([goal_eef_vel.time for goal_eef_vel in goal_eef_vels])
csv_goal_eef_vel_times = csv_goal_eef_vel_times.reshape(-1, 1)
csv_goal_eef_vels = np.concatenate((csv_goal_eef_vel_times, csv_goal_eef_velocities), axis=1)

with open(joint_states_path, mode='w') as csv_file:
    csvwriter = csv.writer(csv_file)
    csvwriter.writerows(csv_joint_states)

with open(motor_states_path, mode='w') as csv_file:
    csvwriter = csv.writer(csv_file)
    csvwriter.writerows(csv_motor_states)

with open(goal_eef_vel_path, mode='w') as csv_file:
    csvwriter = csv.writer(csv_file)
    csvwriter.writerows(csv_goal_eef_vels)




