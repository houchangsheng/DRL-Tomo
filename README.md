# DRL-Tomo
- DRL-Tomo: Deep Reinforcement Learning-based Approach to Augmented Data Generation for Network Tomography

## DRL-Tomo
- Network tomography tool

### Files
- PPO: the implementation of deep reinforcement learning algorithm PPO (Proximal Policy Optimization)
- rocketfuel-weights: the public dataset from the Rocketfuel project, https://research.cs.washington.edu/networking/rocketfuel/
- DataPro.py: the data pre-processing procedure
- DRLTomo.py: the implementation of DRL-Tomo

### Compile and Run
- Run the DRL-Tomo
```
$ python DRLTomo.py
```

## PPO
- the implementation of deep reinforcement learning algorithm PPO (Proximal Policy Optimization)

### Files
- agent.py: execution of PPO algorithm
- env.py: build an environment for deep reinforcement learning
- memory.py: build buffers for training data
- model.py: the neural network model used by the PPO algorithm
- train.py: train and test the model
- utils.py: define functions of "make_dir", "save_results", etc.
