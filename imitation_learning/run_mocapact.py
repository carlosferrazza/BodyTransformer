import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
from collections import deque
import torch
from tqdm import tqdm
from datetime import datetime
import random
from models.networks import BodyNet, Transformer, MLP, SoftBiasTransformer, BodyTransformer
from absl import app, flags
import wandb
from utils.mocapact_utils import HDF5Dataset, MocapactWrapper
from dm_control.locomotion.tasks.reference_pose import types

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', None, 'Directory where the data is stored', required=True)
flags.DEFINE_string('entity', None, 'Wandb entity')

flags.DEFINE_integer('seed', 100, 'Random seed.')
flags.DEFINE_integer('nepochs', 100, 'Number of epochs to train the model')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_every', 5, 'Number of epochs to evaluate the model')

flags.DEFINE_float('lr', 1e-4, 'Learning rate')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')

flags.DEFINE_string('network_type', 'body_transformer', 'Type of network to use, can be [mlp, transformer, soft_bias_transformer, body_transformer]')
flags.DEFINE_integer('nlayers', 16, 'Number of transformer layers')
flags.DEFINE_integer('embedding_dim', 64, 'Dimension of the embeddings')
flags.DEFINE_integer('dim_feedforward', 1024, 'Dimension of the feedforward network layer')
flags.DEFINE_integer('nheads', 5, 'Number of heads in the transformer layer')
flags.DEFINE_string('use_positional_encoding', 'False', 'Whether to use positional encoding')
flags.DEFINE_string('is_mixed', "False", 'Whether to interleave masked and unmasked attention layers')
flags.DEFINE_integer('first_hard_layer', 0, 'Index of the first masked layer for the bot-mix')
flags.DEFINE_string('use_stochastic_policy', 'False', 'Whether to use a stochastic policy')

def run_policy(net, env, device='cpu', mean_inputs=None, std_inputs=None):
    net.eval()

    obs = env.reset()
    max_steps = env.max_steps

    n_cameras = 1

    returns = []
    lengths = []
    returns_norm = []
    lengths_norm = []
    max_lengths = []
    ret = 0
    ln = 0
    ep = 0
    frames = []
    while ep < FLAGS.eval_episodes:
        obs = torch.from_numpy(obs).float().to(device)

        if mean_inputs is not None and std_inputs is not None:
            obs = (obs - mean_inputs.reshape(-1)) / (std_inputs.reshape(-1) + 1e-8)

        if ep == 0:
            frames.append(np.concatenate([env.render(mode='rgb_array') for i in range(n_cameras)], axis=1))  # [H, W, C]

        action = net.mode(obs.unsqueeze(0)).squeeze().detach().cpu().numpy()
        action = np.clip(action, -1, 1)

        next_obs, reward, done, info = env.step(action)
        ret += reward
        ln += 1

        obs = next_obs

        if done or ln >= max_steps:
            print("Episode {} return: {} length: {} max_length: {}".format(ep, ret, ln, max_steps))
            returns.append(ret)
            lengths.append(ln)
            max_lengths.append(max_steps)
            
            ret = 0
            ln = 0
            obs = env.reset()
            max_steps = env.max_steps
            
            ep += 1

    frames = np.array(frames)  # [T, H, W, C]
    frames = np.transpose(frames, (0, 3, 1, 2))  # [T, C, H, W]

    returns_norm = np.array(returns) / np.array(max_lengths)
    print("Mean return normalized: ", np.mean(returns_norm))
    print("Std return normalized: ", np.std(returns_norm))

    lengths_norm = np.array(lengths) / np.array(max_lengths)
    print("Mean length normalized: ", np.mean(lengths_norm))
    print("Std length normalized: ", np.std(lengths_norm))

    net.train()

    return np.mean(returns_norm), np.std(returns_norm), np.mean(lengths_norm), np.std(lengths_norm), frames


def train(net, trainloader, optimizer, criterion, train_env=None, test_env=None, device='cpu', logger=None, mean_inputs=None, std_inputs=None, use_stochastic_policy=False):

    # Save model in wandb folder
    if logger is not None:
        # make new subdirectory for models
        os.makedirs(os.path.join(wandb.run.dir, 'models'), exist_ok=True)

    for epoch in range(FLAGS.nepochs):  # loop over the dataset multiple times

        running_loss = 0.0
        pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader))
        for i, data in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.reshape(-1, inputs.shape[1:].numel())
            if not use_stochastic_policy:
                outputs = net.mode(inputs)
                loss = criterion(outputs, labels)
            else:
                loss = -net.log_prob(inputs, labels).mean()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss
            if i % 100 == 99:    # print every 2000 mini-batches
                avg_loss = running_loss / i
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')

            pbar.set_description(f'Training epoch {epoch + 1}')


        
        if logger is not None:
            logger.log({'loss': avg_loss}, step=epoch)
            
        # Save model in wandb folder
        if logger is not None and (epoch + 1) % FLAGS.eval_every == 0:
            torch.save(net.state_dict(), os.path.join(wandb.run.dir, f'models/model_{epoch}.pt'))

        if (epoch + 1) % FLAGS.eval_every == 0:

            mean_norm, std_norm, mean_len_norm, std_len_norm, frames = run_policy(net, train_env, device=device, mean_inputs=mean_inputs, std_inputs=std_inputs)

            if logger is not None:
                logger.log({'train/mean_return_norm': mean_norm,
                            'train/std_return_norm': std_norm,
                            'train/mean_length_norm': mean_len_norm,
                            'train/std_length_norm': std_len_norm
                            }, step=epoch)
                logger.log({
                    'train/video/video': wandb.Video(frames, fps=30, format='mp4')
                }, step=epoch)

            mean_norm, std_norm, mean_len_norm, std_len_norm, frames = run_policy(net, test_env, device=device, mean_inputs=mean_inputs, std_inputs=std_inputs)
            if logger is not None:
                logger.log({'test/mean_return_norm': mean_norm,
                            'test/std_return_norm': std_norm,
                            'test/mean_length_norm': mean_len_norm,
                            'test/std_length_norm': std_len_norm
                            }, step=epoch)
                logger.log({
                    'test/video/video': wandb.Video(frames, fps=30, format='mp4')
                }, step=epoch)


def main(_):

    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    env_name = 'mocapact'

    dir = FLAGS.data_dir

    train_dataset = HDF5Dataset(f'{dir}/dataset_split.hdf5', test=False)
    test_dataset = HDF5Dataset(f'{dir}/dataset_split.hdf5', test=True)

    train_clip_id = [s.decode("utf-8") for s in train_dataset.get_names()]
    test_clip_id = [s.decode("utf-8") for s in test_dataset.get_names()]

    run_name = f"{env_name}_bc_seed_{FLAGS.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(entity=FLAGS.entity, project='mocapact-bc', name=run_name, config=FLAGS)

    ref_steps = [1, 2, 3, 4, 5]
    task_kwargs = dict(
        reward_type="comic",
        ghost_offset=np.array([1.0, 0.0, 0.0]),
        always_init_at_clip_start=True,
    )
    env_kwargs = dict(
        dataset=types.ClipCollection(ids=train_clip_id),
        ref_steps=ref_steps,
        act_noise=0.0,
        task_kwargs=task_kwargs,
        include_clip_id=True,
    )

    train_env = MocapactWrapper(env_name, train_dataset.get_observable_indices(), dir, use_task_id=False, **env_kwargs)

    env_kwargs['dataset'] = types.ClipCollection(ids=test_clip_id)

    test_env = MocapactWrapper(env_name, test_dataset.get_observable_indices(), dir,  use_task_id=False, **env_kwargs)

    # normalize inputs
    mean_inputs, std_inputs = train_dataset.get_stats()
    
    print("train_dataset", len(train_dataset), flush=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=4
    )

    device = 'cuda'
    
    embedding_dim = FLAGS.embedding_dim
    nheads = FLAGS.nheads

    dim_feedforward = FLAGS.dim_feedforward
    
    # Fixed for mocapact
    nbodies = 32
    action_dim = 56

    is_mixed = False if FLAGS.is_mixed == "False" else True
    use_positional_encoding = False if FLAGS.use_positional_encoding == "False" else True

    if FLAGS.network_type == 'mlp':
        net = MLP(embedding_dim * nbodies, hidden_sizes=(dim_feedforward,dim_feedforward))
    elif FLAGS.network_type == 'transformer':
        embedding_dim *= nheads
        net = Transformer(nbodies, embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, nlayers=FLAGS.nlayers, use_positional_encoding=use_positional_encoding)
    elif FLAGS.network_type == 'soft_bias_transformer':
        embedding_dim *= nheads
        net = SoftBiasTransformer(nbodies, env_name, embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, nlayers=FLAGS.nlayers)
    elif FLAGS.network_type == 'body_transformer':
        embedding_dim *= nheads
        net = BodyTransformer(nbodies, env_name, embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, num_layers=FLAGS.nlayers, is_mixed=is_mixed, use_positional_encoding=use_positional_encoding, first_hard_layer=FLAGS.first_hard_layer)

    model = BodyNet(env_name, net, action_dim=action_dim, embedding_dim=embedding_dim, global_input=FLAGS.network_type=='mlp', device=device)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("training on", device)

    model.to(device)

    mean_inputs = mean_inputs.to(device)
    std_inputs = std_inputs.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr)
    
    criterion = torch.nn.MSELoss()

    train(model, train_dataloader, optimizer, criterion, train_env=train_env, test_env=test_env, device=device, logger=wandb, mean_inputs=mean_inputs, std_inputs=std_inputs)

    print("Done")


if __name__ == '__main__':
    
    app.run(main)