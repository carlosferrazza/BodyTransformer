#!/home/unitree/agility_ziwenz_venv/bin/python
import os
import os.path as osp
import json
import numpy as np
import torch
from collections import OrderedDict
from functools import partial
from typing import Tuple

from a1_real_noros import UnitreeA1Real, resize2d
from rsl_rl import modules
from rsl_rl.utils.utils import get_obs_slice

import time
import sys


class StandOnlyModel(torch.nn.Module):
    def __init__(self, action_scale, dof_pos_scale, tolerance= 0.2, delta= 0.1):
        print("Using stand only model, please make sure the proprioception is 48 dim.")
        print("Using stand only model, -36 to -24 must be joint position.")
        super().__init__()
        if isinstance(action_scale, (tuple, list)):
            self.register_buffer("action_scale", torch.tensor(action_scale))
        else:
            self.action_scale = action_scale
        if isinstance(dof_pos_scale, (tuple, list)):
            self.register_buffer("dof_pos_scale", torch.tensor(dof_pos_scale))
        else:
            self.dof_pos_scale = dof_pos_scale
        self.tolerance = tolerance
        self.delta = delta

    def forward(self, obs):
        joint_positions = obs[..., -36:-24] / self.dof_pos_scale
        diff_large_mask = torch.abs(joint_positions) > self.tolerance
        target_positions = torch.zeros_like(joint_positions)
        target_positions[diff_large_mask] = joint_positions[diff_large_mask] - self.delta * torch.sign(joint_positions[diff_large_mask])
        return torch.clip(
            target_positions / self.action_scale,
            -1.0, 1.0,
        )
    
    def reset(self, *args, **kwargs):
        pass

def load_walk_policy(env, model_dir):
    """ Load the walk policy from the model directory """
    if model_dir == None:
        model = StandOnlyModel(
            action_scale= env.action_scale,
            dof_pos_scale= env.obs_scales["dof_pos"],
        )
        policy = torch.jit.script(model)

    else:
        with open(osp.join(model_dir, "config.json"), "r") as f:
            config_dict = json.load(f, object_pairs_hook= OrderedDict)
        obs_components = config_dict["env"]["obs_components"]
        privileged_obs_components = config_dict["env"].get("privileged_obs_components", obs_components)
        model = getattr(modules, config_dict["runner"]["policy_class_name"])(
            num_actor_obs= env.get_num_obs_from_components(obs_components),
            num_critic_obs= env.get_num_obs_from_components(privileged_obs_components),
            num_actions= 12,
            **config_dict["policy"],
        )
        model_names = [i for i in os.listdir(model_dir) if i.startswith("model_")]
        model_names.sort(key= lambda x: int(x.split("_")[-1].split(".")[0]))
        state_dict = torch.load(osp.join(model_dir, model_names[-1]), map_location= "cpu")
        model.load_state_dict(state_dict["model_state_dict"])
        # model.to("cuda")
        model.eval()
        model_action_scale = torch.tensor(config_dict["control"]["action_scale"]) if isinstance(config_dict["control"]["action_scale"], (tuple, list)) else torch.tensor([config_dict["control"]["action_scale"]])[0]
        if not (torch.is_tensor(model_action_scale) and (model_action_scale == env.action_scale).all()):
            action_rescale_ratio = model_action_scale / env.action_scale
            print("walk_policy action scaling:", action_rescale_ratio.tolist())
        else:
            action_rescale_ratio = 1.0
        # memory_module = model.memory_a
        # memory_module.to("cuda")
        actor_mlp = model.actor
        @torch.jit.script
        def policy_run(obs):
            actions = actor_mlp(obs)
            # recurrent_embedding = memory_module(obs)
            # actions = actor_mlp(recurrent_embedding.squeeze(0))
            return actions
        if (torch.is_tensor(action_rescale_ratio) and (action_rescale_ratio == 1.).all()) \
            or (not torch.is_tensor(action_rescale_ratio) and action_rescale_ratio == 1.):
            policy = policy_run
        else:
            policy = lambda x: policy_run(x) * action_rescale_ratio
    
    return policy, model

def standup_procedure(env, ros_rate, angle_tolerance= 0.1,
        kp= None,
        kd= None,
        warmup_timesteps= 25,
        device= "cpu",
    ):
    """
    Args:
        warmup_timesteps: the number of timesteps to linearly increase the target position
    """
    print("Robot standing up, please wait ...")

    target_pos = torch.zeros((1, 12), device= device, dtype= torch.float32)
    standup_timestep_i = 0
    
    env.standup()

    print("Robot standing up procedure finished!")

def main(args):
    
    """ Not finished this modification yet """
    # if args.logdir is not None:
    #     rospy.loginfo("Use logdir/config.json to initialize env proxy.")
    #     with open(osp.join(args.logdir, "config.json"), "r") as f:
    #         config_dict = json.load(f, object_pairs_hook= OrderedDict)
    # else:
    #     assert args.walkdir is not None, "You must provide at least a --logdir or --walkdir"
    #     rospy.logwarn("You did not provide logdir, use walkdir/config.json for initializing env proxy.")
    #     with open(osp.join(args.walkdir, "config.json"), "r") as f:
    #         config_dict = json.load(f, object_pairs_hook= OrderedDict)
    assert args.logdir is not None or args.walkdir is not None, "You must provide at least a --logdir or --walkdir"

    with open(osp.join(args.walkdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
    
    duration = config_dict["sim"]["dt"] * config_dict["control"]["decimation"] # in sec
    # config_dict["control"]["stiffness"]["joint"] -= 2.5 # kp

    model_device = torch.device("cpu") if args.mode == "upboard" else torch.device("cuda")

    unitree_real_env = UnitreeA1Real(
        robot_namespace= args.namespace,
        cfg= config_dict,
        # extra_cfg= dict(
        #     motor_strength= torch.tensor([
        #         1., 1./0.9, 1./0.9,
        #         1., 1./0.9, 1./0.9,
        #         1., 1., 1.,
        #         1., 1., 1.,
        #     ], dtype= torch.float32, device= model_device, requires_grad= False),
        # ),
    )

    model = getattr(modules, config_dict["runner"]["policy_class_name"])(
        num_actor_obs= unitree_real_env.num_obs,
        num_critic_obs= unitree_real_env.num_privileged_obs,
        num_actions= 12,
        obs_segments= unitree_real_env.obs_segments,
        privileged_obs_segments= unitree_real_env.privileged_obs_segments,
        **config_dict["policy"],
    )
    config_dict["terrain"]["measure_heights"] = False
    # load the model with the latest checkpoint

    print("duration: {}, motor Kp: {}, motor Kd: {}".format(
        duration,
        config_dict["control"]["stiffness"]["joint"],
        config_dict["control"]["damping"]["joint"],
    ))
    # rospy.loginfo("[Env] motor strength: {}".format(unitree_real_env.motor_strength))

    # extract and build the torch ScriptFunction
    actor_mlp = model.actor
    @torch.jit.script
    def policy(obs):
        actions = actor_mlp(obs)
        return actions
    
    walk_policy, walk_model = load_walk_policy(unitree_real_env, args.walkdir)

    unitree_real_env.start_noros()
    time.sleep(5.0)
    
    rate = duration
    with torch.no_grad():
        # if not args.debug:
        #     standup_procedure(unitree_real_env, rate,
        #         angle_tolerance= 0.2,
        #         kp= 40,
        #         kd= 0.5,
        #         warmup_timesteps= 50,
        #         device= model_device,
        #     )
        walk_time = []
        first_run = True
        while True:
            # inference_start_time = rospy.get_time()
            # check remote controller and decide which policy to use
            
            walk_model.reset()

            # unitree_real_env.update_low_state()
            
            # tic = time.time()
            walk_obs = unitree_real_env._get_proprioception_obs()#.to("cuda")
            
            # print("walk_obs: ", walk_obs)
            # sys.exit(0)
            actions = walk_policy(walk_obs).to("cpu")
            
            unitree_real_env.send_action(actions)
            
            # if not first_run:
            #     time.sleep(rate - (toc - tic))
            # else:
            #     first_run = False

if __name__ == "__main__":
    """ The script to run the A1 script in ROS.
    It's designed as a main function and not designed to be a scalable code.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace",
        type= str,
        default= "/a112138",                    
    )
    parser.add_argument("--logdir",
        type= str,
        help= "The log directory of the trained model",
        default= None,
    )
    parser.add_argument("--walkdir",
        type= str,
        help= "The log directory of the walking model, not for the skills.",
        default= None,
    )
    parser.add_argument("--mode",
        type= str,
        help= "The mode to determine which computer to run on.",
        choices= ["jetson", "upboard", "full"],                
    )
    parser.add_argument("--debug",
        action= "store_true",
    )

    args = parser.parse_args()
    main(args)