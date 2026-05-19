# Combining Exploration and Imitation in Contact-rich Task Learning on an Articulated Soft Robot Arm

## Setup Simulation Environment
- Clone robosuite repository
  ```
  git clone https://github.com/LauEls/robosuite.git
  ```
- Install requirements
  ```
  cd robosuite
  pip3 install -r requirements.txt
  ```

## Rollout trained model
- 

## Evaluation Experiments in Simulation

### Ablation Study
|                      Experiment Reference                      | Results |         |         |         |        Dataset        |
|:--------------------------------------------------------------:|:-------:|:-------:|:-------:|:-------:|:---------------------:|
| Comparing number of demonstrations in the demonstration buffer |    [5](/runs/door_mirror/gh360/osc_pose/online/5_demos/)    |    [10](/runs/door_mirror/gh360/osc_pose/online/10_demos/)   |    [20](/runs/door_mirror/gh360/osc_pose/online/20_demos/)   |    [50](/runs/door_mirror/gh360/osc_pose/online/50_demos/)   | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |
| Percentage of Demonstration Data in Learning Batch             |   [10%](/runs/door_mirror/gh360/osc_pose/online/10_percent_demo_ratio/)   |   [25%](/runs/door_mirror/gh360/osc_pose/online/25_percent_demo_ratio/)   |   [50%](/runs/door_mirror/gh360/osc_pose/online/20_demos/)   |   [75%](/runs/door_mirror/gh360/osc_pose/online/75_percent_demo_ratio/)   | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |

### Offline Learning
|   Experiment Result   |        Dataset        |
|:---------------------:|:---------------------:|
| [Expert](/runs/door_mirror/gh360/osc_pose/offline/expert_dataset/)                | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |
| [Random Expert](runs/door_mirror/gh360/osc_pose/offline/random_expert_dataset)         | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1_random.npy) |
| [Gradual Random Expert](/runs/door_mirror/gh360/osc_pose/offline/gradual_random_expert_dataset/) | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1_gradual_random.npy) |

### Final Evaluation Results
#### Full Length
| Model and Results | Dataset |
|:-----------------:|:-------:|
| [TD7 + Demo](/runs/door_mirror/gh360/osc_pose/online/td7_with_demos_full_length/) | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |
| [TD7](/runs/door_mirror/gh360/osc_pose/online/td7_full_length/) |  |
| [TD7 Offline](/runs/door_mirror/gh360/osc_pose/offline/gradual_random_expert_dataset_full_length/) | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1_gradual_random.npy) |
| Demonstrations | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |

#### First 200 Episodes
| Model and Results | Dataset |
|:-----------------:|:-------:|
| [TD7 + Demo](/runs/door_mirror/gh360/osc_pose/online/20_demos/) | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |
| [TD7](/runs/door_mirror/gh360/osc_pose/online/td7/) |  |
| [TD7 Offline](/runs/door_mirror/gh360/osc_pose/offline/gradual_random_expert_dataset/) | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1_gradual_random.npy) |
| Demonstrations | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |

## Evaluation Experiments on Real GH360 Robot
| Model and Results | Dataset | ROSBags |
|:-----------------:|:-------:|:-------:|
| [TD7 + Demo](/runs/door/real_gh360/eef_vel/online/td7_with_demos/) | [demonstration_dataset](/demonstrations/gh360_door_demonstration_v8.npy) | [recording]() |
| [TD7](/runs/door/real_gh360/eef_vel/online/td7/) |  |  |
| [BC](/runs/door/real_gh360/eef_vel/online/bc/) | [demonstration_dataset](/demonstrations/gh360_door_demonstration_v8.npy) | [recording]() |
| Demonstrations | [demonstration_dataset](/demonstrations/gh360_door_demonstration_v8.npy) | [recording]() |