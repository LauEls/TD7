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





|                      Experiment Reference                      | Results |         |         |         |        Dataset        |
|:--------------------------------------------------------------:|:-------:|:-------:|:-------:|:-------:|:---------------------:|
| Comparing number of demonstrations in the demonstration buffer |    [5](/runs/door_mirror/gh360/osc_pose/online/5_demos/)    |    [10]()   |    [20]()   |    [50]()   | demonstration_dataset |
| Percentage of Demonstration Data in Learning Batch             |   10%   |   20%   |   50%   |   75%   | demonstration_dataset |