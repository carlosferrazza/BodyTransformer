import numpy as np
import pickle 
import h5py
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import h5py
from mocapact.envs import tracking
from models import mappings 
import sys

# clip: CMU_009_12
# snippet: CMU_009_12-0-198
# rollout: CMU_009_12-0-198/0

class MocapactWrapper(tracking.MocapTrackingGymEnv):

    def __init__(self, env_name, observable_indices, dir, use_task_id, **kwargs):

        super().__init__(**kwargs)

        self.observable_indices = observable_indices
        self.env_name = env_name
        self.ref_steps = kwargs['ref_steps']
        self.dir = dir
        self.use_task_id = use_task_id

        self.max_steps = None

    def step(self, action):
        obs, reward, done, info = super().step(action)

        obs = self.process_obs(obs)

        return obs, reward, done, info
    
    def process_obs(self, obs):
        new_obs = np.zeros(mappings.DIMS[self.env_name])
        for key in obs.keys():
            key = key.split('/')[-1]
            if key == 'clip_id':
                if self.use_task_id:
                    new_obs[-1] = obs["walker/clip_id"].item()
            else:
                new_obs[self.observable_indices[key]] = obs[f"walker/{key}"]

        return new_obs

    def reset(self):
        obs = super().reset()
        clip_name = self._dataset.ids[obs["walker/clip_id"].item()]
        with h5py.File(f"{self.dir}/{clip_name}.hdf5") as f:
            for key in f.keys():
                if key.startswith("CMU"):
                    start, end = key.split('-')[-2:]
                    self.max_steps = int(end) - int(start) - len(self.ref_steps) - 1
                    break
        return self.process_obs(obs)

class HDF5Dataset(Dataset):
    def __init__(self, file_path, normalize=True, test=False):
        self.file_path = file_path
        self.length = None
        self.normalize = normalize

        
        if test == False: 
            self.input_field = 'train_inputs'
            self.target_field = 'train_targets'
            self.name_field = 'train_names'
        else:
            self.input_field = 'val_inputs'
            self.target_field = 'val_targets'
            self.name_field = 'val_names'

        with h5py.File(self.file_path, 'r') as hf:
            for gname, group in hf.items():
                if gname == self.input_field:
                    self.length = len(group)

        self._open_hdf5()


        self.mean, self.std = self.get_stats()

    def __len__(self):
        assert self.length is not None
        return self.length

    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, index):
        
        x = self._hf[self.input_field][index]
        y = self._hf[self.target_field][index]
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.normalize:
            x = (x - self.mean) / (self.std + 1e-8)

        return (x, y)
    
    def get_stats(self):
        
        mean, std = self._hf['inputs_mean'][:], self._hf['inputs_std'][:]

        return torch.from_numpy(mean).float(), torch.from_numpy(std).float()

    def get_names(self):

        return self._hf[self.name_field]
    
    def get_observable_indices(self):

        return self._hf['observable_indices']

def load_mocapact_data(folder, type='obs'):
    
    train_clips = np.loadtxt('utils/train_clips.txt', dtype=str)
    val_clips = np.loadtxt('utils/val_clips.txt', dtype=str)

    current_clip = 0

    train_obs = []
    train_actions = []
    train_tasks = []
    train_names = []
    val_obs = []
    val_actions = []
    val_tasks = []
    val_names = []
    for file in tqdm(sorted(os.listdir(folder))): # for each clip
        if file.endswith('.hdf5') and file.startswith('CMU'):
            
            data = os.path.join(folder, file)

            with h5py.File(data, 'r') as f:
                
                if file.split('.')[0] in train_clips:
                    all_obs = train_obs
                    all_actions = train_actions
                    all_tasks = train_tasks
                    all_names = train_names
                elif file.split('.')[0] in val_clips:
                    all_obs = val_obs
                    all_actions = val_actions
                    all_tasks = val_tasks
                    all_names = val_names
                else:
                    raise ValueError("Clip not in train or val set")

                if type == 'names':
                    all_names.append(file.split('.')[0])

                S = f['n_start_rollouts'][()]
                R = f['n_rsi_rollouts'][()]
                for key in f.keys(): # for each snippet
                    if key.startswith('CMU'):
                        for i in range(S+R): # for each rollout
                            if f[key]['early_termination'][i]:
                                continue
                            if type == 'obs':
                                all_obs.append(f[key][str(i)]['observations']['proprioceptive'][:-1]) 
                            elif type == 'actions':
                                all_actions.append(f[key][str(i)]['actions'][:])
                            elif type == 'tasks':
                                all_tasks.append(np.ones((f[key][str(i)]['observations']['proprioceptive'][:-1].shape[0], 1)) * current_clip)
                current_clip += 1

    if type=='obs':
        train_obs = np.concatenate(train_obs, axis=0)
        print("train_obs shape: ", train_obs.shape)
        pickle.dump(train_obs, open(os.path.join(folder, 'train_obs.pkl'), 'wb'), protocol=5)

        val_obs = np.concatenate(val_obs, axis=0)
        print("val_obs shape: ", val_obs.shape)
        pickle.dump(val_obs, open(os.path.join(folder, 'val_obs.pkl'), 'wb'))

    elif type=='actions':
        train_actions = np.concatenate(train_actions, axis=0)
        print("train_actions shape: ", train_actions.shape)
        pickle.dump(train_actions, open(os.path.join(folder, 'train_actions.pkl'), 'wb'))

        val_actions = np.concatenate(val_actions, axis=0)
        print("val_actions shape: ", val_actions.shape)
        pickle.dump(val_actions, open(os.path.join(folder, 'val_actions.pkl'), 'wb'))

    elif type=='tasks':
        train_tasks = np.concatenate(train_tasks, axis=0)
        print("train_tasks shape: ", train_tasks.shape)
        pickle.dump(train_tasks, open(os.path.join(folder, 'train_tasks.pkl'), 'wb'))

        val_tasks = np.concatenate(val_tasks, axis=0)
        print("val_tasks shape: ", val_tasks.shape)
        pickle.dump(val_tasks, open(os.path.join(folder, 'val_tasks.pkl'), 'wb'))

    elif type=='names':
        np.savetxt(os.path.join(folder, 'train_names.txt'), train_names, fmt='%s')
        np.savetxt(os.path.join(folder, 'val_names.txt'), val_names, fmt='%s')

def generate_files(folder):

    print("Generating files... hold on even after 100%% has been reached")
    load_mocapact_data(folder, type='obs')
    print("Generated obs")
    load_mocapact_data(folder, type='actions')
    print("Generated actions")
    load_mocapact_data(folder, type='tasks')
    print("Generated tasks")
    load_mocapact_data(folder, type='names')
    print("Generated names")


def create_hdf5_split(dir):
    
    indices_dict = h5py.File(os.path.join(dir, 'CMU_001_01.hdf5'), 'r')['observable_indices']
    hf = h5py.File(os.path.join(dir, 'dataset_split.hdf5'), 'w')
    indices_dict.copy('walker', hf, name='observable_indices')
    print("keys: ", hf['observable_indices'].keys())
    print("done loading indices", flush=True)

    inputs = pickle.load(open(os.path.join(dir, 'train_obs.pkl'),'rb'))
    targets = pickle.load(open(os.path.join(dir, 'train_actions.pkl'),'rb'))
    names = np.loadtxt(os.path.join(dir, 'train_names.txt'), dtype=str) 
    mean = np.mean(inputs, axis=0)
    std = np.std(inputs, axis=0) 

    # transform names to list of strings
    names = names.tolist()
    print("names: ", names)

    hf['/inputs_mean'] = mean
    hf['/inputs_std'] = std
    hf['/train_inputs'] = inputs
    hf['/train_targets'] = targets
    hf['/train_names'] = names

    inputs = pickle.load(open(os.path.join(dir, 'val_obs.pkl'),'rb'))
    targets = pickle.load(open(os.path.join(dir, 'val_actions.pkl'),'rb'))
    names = np.loadtxt(os.path.join(dir, 'val_names.txt'), dtype=str)

    # transform names to list of strings
    names = names.tolist()
    print("names: ", names)
    
    hf['/val_inputs'] = inputs
    hf['/val_targets'] = targets
    hf['/val_names'] = names

    hf.close()

if __name__ == "__main__":

    dir = sys.argv[1]

    generate_files(dir)
    print("Creating dataset split... might take some time")
    create_hdf5_split(dir)

    dataset = HDF5Dataset(f'{dir}/dataset_split.hdf5')
    train_loader = DataLoader(dataset, batch_size=32,
                                shuffle=True, num_workers=0)
    print("next shape: ", next(iter(train_loader))[0].shape)
    print("names: ", dataset.get_names()[()])

    print("done")

                            
