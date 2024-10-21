# Reinforcement Learning Instructions

This code is based off the [rl_games](https://github.com/Denys88/rl_games) and [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) codebases.

## Install the simulation environment

Install IsaacGym by following the instructions in the [official page](https://developer.nvidia.com/isaac-gym).
Then, install the required packages:
```
pip install -e IsaacGymEnvs
pip install -e rl_games
```

## Train the policy

To train the policy, run the following commands:
```
cd IsaacGymEnvs/isaacgymenvs
python launch.py
```
See the `launch.py` file for the available options.

