import torch

MAPS = { # First index is a list that contains the indices of the observation, the second index is a list that contains the indices of the action 
    'a1': {
        'root': [list(range(0,12)), []],
        'FL_hip': [[12, 24, 36], [0]],
        'FL_thigh': [[13, 25, 37], [1]],
        'FL_calf': [[14, 26, 38], [2]],
        'FR_hip': [[15, 27, 39], [3]],
        'FR_thigh': [[16, 28, 40], [4]],
        'FR_calf': [[17, 29, 41], [5]],
        'RL_hip': [[18, 30, 42], [6]],
        'RL_thigh': [[19, 31, 43], [7]],
        'RL_calf': [[20, 32, 44], [8]],
        'RR_hip': [[21, 33, 45], [9]],
        'RR_thigh': [[22, 34, 46], [10]],
        'RR_calf': [[23, 35, 47], [11]],
    },
}

SP_MATRICES = {
    'a1': torch.Tensor(
        [[0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        [1, 0, 1, 2, 2, 3, 4, 2, 3, 4, 2, 3, 4],
        [2, 1, 0, 1, 3, 4, 5, 3, 4, 5, 3, 4, 5],
        [3, 2, 1, 0, 4, 5, 6, 4, 5, 6, 4, 5, 6],
        [1, 2, 3, 4, 0, 1, 2, 2, 3, 4, 2, 3, 4],
        [2, 3, 4, 5, 1, 0, 1, 3, 4, 5, 3, 4, 5],
        [3, 4, 5, 6, 2, 1, 0, 4, 5, 6, 4, 5, 6],
        [1, 2, 3, 4, 2, 3, 4, 0, 1, 2, 2, 3, 4],
        [2, 3, 4, 5, 3, 4, 5, 1, 0, 1, 3, 4, 5],
        [3, 4, 5, 6, 4, 5, 6, 2, 1, 0, 4, 5, 6],
        [1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 0, 1, 2],
        [2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 1, 0, 1],
        [3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 2, 1, 0]]
    ),
}

DIMS = {
    'a1': 48,
}

def is_map_empty(map):
    for k, v in map.items():
        if len(v[0]) or len(v[1]):
            return False
    return True

class Mapping:

    def __init__(self, env_name, mask_empty=False):
        
        self.dim = DIMS[env_name]
        
        self.map = MAPS[env_name].copy()
        self.shortest_path_matrix = SP_MATRICES[env_name]

        if mask_empty:
            mask_indices = []
            mask_keys = []
            for i, k, v in zip(range(len(self.map)), self.map.keys(), self.map.values()):
                if len(v[0]) == 0 and len(v[1]) == 0:
                    mask_indices.append(i)
                    mask_keys.append(k)
            
            # remove keys that are empty
            for key in mask_keys:
                self.map.pop(key)
                
            keep_indices = [i for i in range(self.shortest_path_matrix.shape[0]) if i not in mask_indices]
            self.shortest_path_matrix = self.shortest_path_matrix[keep_indices][:,keep_indices]

    def get_map(self):
        return self.map
    
    def create_observation(self, obs, stack_time=True):
        new_obs = {}
        for k, v in self.map.items():
            new_obs[k] = obs[:,v[0]]
            if stack_time:
                new_obs[k] = new_obs[k].reshape(obs.shape[0], -1)
        return new_obs
        