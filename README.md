# For SALE: State-Action Representation Learning for Deep Reinforcement Learning

Official implementation of the TD7 algorithm. 

### Usage

Example online RL:
```
python main.py --env HalfCheetah-v4
```
Example offline RL:
```
python main.py --offline --env halfcheetah-medium-v2 
```

### Software

Results were originally collected with:
- [Gym 0.25.0](https://github.com/openai/gym)
- [MuJoCo 2.3.3](https://github.com/deepmind/mujoco)
- [Pytorch 2.0.0](https://pytorch.org)
- [Python 3.9.13](https://www.python.org)

# Evaluation Experiments in Simulation

## Ablation Study
|                      Experiment Reference                      | Results |         |         |         |        Dataset        |
|:--------------------------------------------------------------:|:-------:|:-------:|:-------:|:-------:|:---------------------:|
| Comparing number of demonstrations in the demonstration buffer |    [5](/runs/door_mirror/gh360/osc_pose/online/5_demos/)    |    [10](/runs/door_mirror/gh360/osc_pose/online/10_demos/)   |    [20](/runs/door_mirror/gh360/osc_pose/online/20_demos/)   |    [50](/runs/door_mirror/gh360/osc_pose/online/50_demos/)   | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |
| Percentage of Demonstration Data in Learning Batch             |   [10%](/runs/door_mirror/gh360/osc_pose/online/10_percent_demo_ratio/)   |   [25%](/runs/door_mirror/gh360/osc_pose/online/25_percent_demo_ratio/)   |   [50%](/runs/door_mirror/gh360/osc_pose/online/20_demos/)   |   [75%](/runs/door_mirror/gh360/osc_pose/online/75_percent_demo_ratio/)   | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |

## Offline Learning
|   Experiment Result   |        Dataset        |
|:---------------------:|:---------------------:|
| [Expert](/runs/door_mirror/gh360/osc_pose/offline/expert_dataset/)                | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |
| [Random Expert](runs/door_mirror/gh360/osc_pose/offline/random_expert_dataset)         | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1_random.npy) |
| [Gradual Random Expert](/runs/door_mirror/gh360/osc_pose/offline/gradual_random_expert_dataset/) | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1_gradual_random.npy) |

## Final Evaluation
### Full Length
| Algorithm | Dataset |
|:---------:|:-------:|
| [TD7 + Demo](/runs/door_mirror/gh360/osc_pose/online/td7_with_demos_full_length/) | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |
| [TD7](/runs/door_mirror/gh360/osc_pose/online/td7_full_length/) |  |
| [TD7 Offline](/runs/door_mirror/gh360/osc_pose/offline/gradual_random_expert_dataset_full_length/) | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1_gradual_random.npy) |
| Demonstrations | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |

### First 200 Episodes
| Algorithm | Dataset |
|:---------:|:-------:|
| [TD7 + Demo](/runs/door_mirror/gh360/osc_pose/online/20_demos/) | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |
| [TD7](/runs/door_mirror/gh360/osc_pose/online/td7/) |  |
| [TD7 Offline](/runs/door_mirror/gh360/osc_pose/offline/gradual_random_expert_dataset/) | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1_gradual_random.npy) |
| Demonstrations | [demonstration_dataset](/demonstrations/gh360_sim_door_demonstration_with_variance_v1.npy) |