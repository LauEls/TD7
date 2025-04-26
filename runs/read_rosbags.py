import rosbag2_py
import numpy as np
import csv
from dataclasses import dataclass
from rclpy.serialization import deserialize_message
from gh360_interfaces.msg import PortStatus
from sensor_msgs.msg import JointState

@dataclass
class ROSBagMsg:
    time: int
    data: any


file_base = 'door/real_gh360/eef_vel/online/v8_corl_with_demos/run_2/'
bag_name = 'rosbag_1745229620'
rosbag_path = file_base + bag_name
joint_states_path = file_base + 'joint_states.csv'
motor_states_path = file_base + 'motor_states.csv'

rosbag_reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py._storage.StorageOptions(
    uri=rosbag_path,
    storage_id='sqlite3')
converter_options = rosbag2_py._storage.ConverterOptions('', '')
rosbag_reader.open(storage_options, converter_options)

joint_states = []
motor_states = []

while rosbag_reader.has_next():
    topic, msg, t = rosbag_reader.read_next()

    if topic.endswith('/gh360/joint_states'):
        msg_dec = deserialize_message(msg, JointState)
        joint_states.append(ROSBagMsg(t, msg_dec))
    
    elif topic.endswith('/gh360/motor_states_sorted'):
        msg_dec = deserialize_message(msg, PortStatus)
        motor_states.append(ROSBagMsg(t, msg_dec))


print(f"Joint states: {len(joint_states)}")
print(f"Motor states: {len(motor_states)}")

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

with open(joint_states_path, mode='w') as csv_file:
    csvwriter = csv.writer(csv_file)
    csvwriter.writerows(csv_joint_states)

with open(motor_states_path, mode='w') as csv_file:
    csvwriter = csv.writer(csv_file)
    csvwriter.writerows(csv_motor_states)




