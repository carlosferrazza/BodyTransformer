BODY_NAMES = ["root", "lhipjoint", "lfemur", "ltibia", "lfoot", "ltoes", "rhipjoint", "rfemur", "rtibia", "rfoot", "rtoes", "lowerback", "upperback", "thorax", "lowerneck", "upperneck", "head", "face", "lclavicle", "lhumerus", "lradius", "lwrist", "lhand", "lfingers", "lthumb", "rclavicle", "rhumerus", "rradius", "rwrist", "rhand", "rfingers", "rthumb"]

CMU_HUMANOID_OBS_IDX = {
    "joint_actuator_orders": {
        "lfemur": [5, 6, 7],
        "ltibia": [25],
        "lfoot": [9, 10],
        "ltoes": [26],
        "rfemur": [30, 31, 32],
        "rtibia": [44],
        "rfoot": [34, 35],
        "rtoes": [45],
        "lowerback": [16, 17, 18],
        "upperback": [50, 51, 52],
        "thorax": [47, 48, 49],
        "lowerneck": [19, 20, 21],
        "upperneck": [53, 54, 55],
        "head": [0, 1, 2],
        "lclavicle": [3, 4],
        "lhumerus": [13, 14, 15],
        "lradius": [22],
        "lwrist": [27],
        "lhand": [11, 12],
        "lfingers": [8],
        "lthumb": [23, 24],
        "rclavicle": [28, 29],
        "rhumerus": [38, 39, 40],
        "rradius": [41],
        "rwrist": [46],
        "rhand": [36, 37],
        "rfingers": [33],
        "rthumb": [42, 43],
    },
    "appendage_pos_order": {
        "lfoot": [9, 10, 11],
        "rfoot": [6, 7, 8],
        "head": [12, 13, 14],
        "lradius": [3, 4, 5],
        "rradius": [0, 1, 2],
    },
    "end_effectors_pos_order": {
        "lfoot": [9, 10, 11],
        "rfoot": [6, 7, 8],
        "lradius": [3, 4, 5],
        "rradius": [0, 1, 2],
    },
    "torque_order": {"lhumerus": [0, 1, 2], "rhumerus": [3, 4, 5]},
    "touch_order": {
        "lfoot": [9],
        "ltoes": [6],
        "rfoot": [8],
        "rtoes": [7],
        "lhand": [0],
        "lfingers": [1],
        "lthumb": [2],
        "rhand": [3],
        "rfingers": [4],
        "rthumb": [5],
    },
    "body_order_pos": {
        body_name: list(range((i - 1) * 3, i * 3))
        for i, body_name in enumerate(BODY_NAMES) if i > 0
    },
    "body_order_pos_ref_steps": {
        body_name: sum(
            [
                list(range((i - 1) * 3 + step_n * 31 * 3, i * 3 + step_n * 31 * 3))
                for step_n in range(5)
            ],
            start=[],
        )
        for i, body_name in enumerate(BODY_NAMES) if i > 0
    },
    "body_order_quat": {
        body_name: list(range((i - 1) * 4, i * 4))
        for i, body_name in enumerate(BODY_NAMES) if i > 0
    },
    "body_order_quat_ref_steps": {
        body_name: sum(
            [
                list(range((i - 1) * 4 + step_n * 31 * 4, i * 4 + step_n * 31 * 4))
                for step_n in range(5)
            ],
            start=[],
        )
        for i, body_name in enumerate(BODY_NAMES) if i > 0
    },
    "head_order": {"head": [0]},
}

CMU_HUMANOID_OBS_IDX["joint_actuator_orders_ref_steps"] = {
    k: sum(
        [[joint_idx + step_n * 56 for joint_idx in v] for step_n in range(5)], start=[]
    )
    for k, v in CMU_HUMANOID_OBS_IDX["joint_actuator_orders"].items()
}

CMU_HUMANOID_OBS_IDX["appendage_pos_order_ref_steps"] = {
    k: sum(
        [[joint_idx + step_n * 15 for joint_idx in v] for step_n in range(5)], start=[]
    )
    for k, v in CMU_HUMANOID_OBS_IDX["appendage_pos_order"].items()
}

CMU_HUMANOID_OBS_IDX_TO_ORDER = {
    "walker/actuator_activation": "joint_actuator_orders",
    "walker/appendages_pos": "appendage_pos_order",
    "walker/body_height": "root_information,1",
    "walker/end_effectors_pos": "end_effectors_pos_order",
    "walker/gyro_anticlockwise_spin": "root_information,1",
    "walker/gyro_backward_roll": "root_information,1",
    "walker/gyro_control": "root_information,3",
    "walker/gyro_rightward_roll": "root_information,1",
    "walker/head_height": "head_order",
    "walker/joints_pos": "joint_actuator_orders",
    "walker/joints_vel": "joint_actuator_orders",
    "walker/joints_vel_control": "joint_actuator_orders",
    "walker/orientation": "root_information,9",
    "walker/position": "root_information,3",
    "walker/reference_appendages_pos": "appendage_pos_order_ref_steps",
    "walker/reference_ego_bodies_quats": "body_order_quat_ref_steps",
    "walker/reference_rel_bodies_pos_global": "body_order_pos_ref_steps",
    "walker/reference_rel_bodies_pos_local": "body_order_pos_ref_steps",
    "walker/reference_rel_bodies_quats": "body_order_quat_ref_steps",
    "walker/reference_rel_joints": "joint_actuator_orders_ref_steps",
    "walker/reference_rel_root_pos_local": "root_information,15",
    "walker/reference_rel_root_quat": "root_information,20",
    "walker/sensors_accelerometer": "root_information,3",
    "walker/sensors_gyro": "root_information,3",
    "walker/sensors_torque": "torque_order",
    "walker/sensors_touch": "touch_order",
    "walker/sensors_velocimeter": "root_information,3",
    "walker/time_in_clip": "root_information,1",
    "walker/torso_xvel": "root_information,1",
    "walker/torso_yvel": "root_information,1",
    "walker/veloc_forward": "root_information,1",
    "walker/veloc_strafe": "root_information,1",
    "walker/veloc_up": "root_information,1",
    "walker/velocimeter_control": "root_information,3",
    "walker/world_zaxis": "root_information,3",
}

CMU_HUMANOID_OBS_IDX_SELECTED = {
    'walker/actuator_activation',
    'walker/appendages_pos',
    'walker/body_height',
    'walker/end_effectors_pos',
    'walker/gyro_control',
    'walker/joints_pos',
    'walker/joints_vel',
    'walker/joints_vel_control',
    'walker/sensors_accelerometer',
    'walker/sensors_gyro',
    'walker/sensors_torque',
    'walker/sensors_touch',
    'walker/sensors_velocimeter',
    'walker/time_in_clip', # TODO: experiment removing this
    'walker/velocimeter_control',
    'walker/world_zaxis',
}

CMU_HUMANOID_OBS_IDX_WITH_REF_SELECTED = {
    'walker/actuator_activation', #
    'walker/body_height', #
    'walker/end_effectors_pos', #
    'walker/gyro_control',
    'walker/joints_pos', #
    'walker/joints_vel', #
    "walker/reference_rel_bodies_pos_local", #
    "walker/reference_rel_bodies_quats", #
    'walker/sensors_gyro', #
    'walker/sensors_torque', #
    'walker/sensors_touch', #
    'walker/sensors_velocimeter', #
    'walker/world_zaxis', #
}

def create_cmu_humanoid_obs_idx(cmu_humanoid_obs_idx):
    
    idx_to_order = CMU_HUMANOID_OBS_IDX_TO_ORDER

    obs_selected = CMU_HUMANOID_OBS_IDX_WITH_REF_SELECTED

    obs_dim_so_far = 0
    cmu_humanoid_obs_idx["root"] = [[], []]
    for obs_idx in idx_to_order:
        if obs_idx in obs_selected:
            skip = False
        else:
            skip = True
        obs_dim_this_idx = 0
        obs_order = idx_to_order[obs_idx]
        if "root_information" in obs_order:
            obs_dim_this_idx = int(obs_order.split(",")[-1])
            if not skip:
                cmu_humanoid_obs_idx["root"][0].extend(
                    list(range(obs_dim_so_far, obs_dim_so_far + obs_dim_this_idx))
                )
        else:
            obs_idx_order = CMU_HUMANOID_OBS_IDX[obs_order]
            for body_name, body_indices in obs_idx_order.items():
                obs_dim_this_idx = max(obs_dim_this_idx, max(body_indices))
                if not skip:
                    cmu_humanoid_obs_idx[body_name][0].extend(
                        [obs_dim_so_far + idx for idx in body_indices]
                    )
            obs_dim_this_idx += 1
        obs_dim_so_far += obs_dim_this_idx

    action_order = CMU_HUMANOID_OBS_IDX["joint_actuator_orders"]

    for body_name, body_indices in action_order.items():
        cmu_humanoid_obs_idx[body_name][1].extend(body_indices)

    return cmu_humanoid_obs_idx
