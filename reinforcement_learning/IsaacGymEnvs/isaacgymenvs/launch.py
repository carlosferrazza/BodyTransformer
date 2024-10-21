import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", help="task name, can be HumanoidBob, HumanoidForces, HumanoidHill", default="HumanoidForces")
parser.add_argument("--method", help="method name, can be mlp, transformer, body_transformer", default="body_transformer")
parser.add_argument("--wandb_entity", help="wandb entity", default="")
parser.add_argument("--seed", help="seed", default=100, type=int)
parser.add_argument("--is_mixed", help="is mixed", default=False, type=bool)
args = parser.parse_args()

PYTHON_BIN = 'python'
task = args.task
seed = args.seed
method = args.method
critic_arch = 'same'
wandb_entity = args.wandb_entity

is_mixed = args.is_mixed

use_trimesh = False
if task == 'HumanoidHill':
    task = 'HumanoidForces'
    use_trimesh = True
    num_layers = 12
else:
    num_layers = 10

train = f'{task}Transformer'

cmd = (
        f"{PYTHON_BIN} train.py "
        f"task={task} "
        f"task.env.stateType=humanoid_with_forces "
        f"headless=True "
        f"capture_video=False "
        f"force_render=False "
        f"sim_device=cuda:0 "
        f"rl_device=cuda:0 "
        f"wandb_activate=True "
        f"wandb_entity={wandb_entity} "
        f"wandb_project=rl-result "
        f"seed={seed} "
        f"train={train} "
        f"train.params.network.name={method} "
        f"train.params.config.max_epochs=10000 "
        f"train.params.config.name={task}_{method}_{seed} "
        f"train.params.config.minibatch_size=8192 "
        f"num_envs=2048 "
        f"train.params.network.transformer.critic_type={critic_arch} "
        f"train.params.network.transformer.dim_embeddings={64 if method == 'mlp' else 64} "
        f"train.params.network.transformer.num_layers={num_layers} "
        f"train.params.network.transformer.num_heads=2 "
        f"train.params.network.transformer.dim_feedforward={150 if method == 'mlp' else 128} "
    )

if use_trimesh:
    cmd += f"task.env.terrain.terrainType=trimesh "

if is_mixed:
    cmd += f"train.params.network.transformer.bias_type=mixed "
else:
    cmd += f"train.params.network.transformer.bias_type=hard "

os.system(cmd)