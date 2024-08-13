# Unitree A1 Sim-to-Real Instructions

## Install the simulation environment

Install IsaacGym by following the instructions in the [official page](https://developer.nvidia.com/isaac-gym).
Then, install the required packages:
```
pip install -e rsl_rl
pip install -e legged_gym
pip install positional_encodings tensorboard
```

## Train the policy

To train the policy, run the following command:
```
python legged_gym/legged_gym/scripts/train.py --headless --task a1_field
```

## Play the policy

To play the policy, run the following command:
```
cd legged_gym
python legged_gym/scripts/play.py --task a1_field --load_run $YOUR_RUN_DIR
```

## Evaluate the policy in the real world

To evaluate the policy in the real world, run the following command:
```
python onboard_codes/a1_noros_run.py --walkdir $YOUR_RUN_DIR
```

## Citation ##
If you use this part of the codebase, please cite the original paper:
```
@inproceedings{
    zhuang2023robot,
    title={Robot Parkour Learning},
    author={Ziwen Zhuang and Zipeng Fu and Jianren Wang and Christopher G Atkeson and S{\"o}ren Schwertfeger and Chelsea Finn and Hang Zhao},
    booktitle={Conference on Robot Learning {CoRL}},
    year={2023}
}
```
