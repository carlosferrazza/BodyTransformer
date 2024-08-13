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
import minari

FLAGS = flags.FLAGS

flags.DEFINE_string('env', 'door-expert-v2', 'Environment name, can be [door-expert-v2, hammer-expert-v2, relocate-expert-v2]')
flags.DEFINE_string('entity', None, 'Wandb entity')

flags.DEFINE_integer('seed', 100, 'Random seed.')
flags.DEFINE_integer('ndemos', 1, 'Number of demonstrations to use for training')
flags.DEFINE_integer('nepochs', 100, 'Number of epochs to train the model')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes used for evaluation.')

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
    
    obs, _ = env.reset()

    n_cameras = 1

    returns = []
    ret = 0
    ep = 0
    success = 0

    env.unwrapped.render_mode = 'rgb_array'
    
    frames = []
    while ep < FLAGS.eval_episodes:
        obs = torch.from_numpy(obs).float().to(device)

        if mean_inputs is not None and std_inputs is not None:
            obs = (obs - mean_inputs.reshape(-1)) / (std_inputs.reshape(-1) + 1e-8)

        if ep == 0:
            frames.append(np.concatenate([env.render() for i in range(n_cameras)], axis=1))  # [H, W, C]

        action = net.mode(obs.unsqueeze(0)).squeeze().detach().cpu().numpy()
        action = np.clip(action, -1, 1)

        next_obs, reward, terminated, truncated, info = env.step(action)
        ret += reward

        done = terminated or truncated

        obs = next_obs

        if done:
            success += info['success']
            obs, _ = env.reset()
            print("Episode {} return: {}".format(ep, ret))
            returns.append(ret)
            ret = 0
            ep += 1

    frames = np.array(frames)  # [T, H, W, C]
    frames = np.transpose(frames, (0, 3, 1, 2))  # [T, C, H, W]

    success_rate = success / FLAGS.eval_episodes

    print("Mean return: ", np.mean(returns))
    print("Std return: ", np.std(returns))
    print("Success rate: ", success_rate)

    net.train()

    return np.mean(returns), np.std(returns), success_rate, frames


def train(net, trainloader, optimizer, criterion, env=None, device='cpu', logger=None, mean_inputs=None, std_inputs=None, use_stochastic_policy=False):

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
            if i % 100 == 99 or i == len(trainloader) - 1:    # print every 100 mini-batches
                avg_loss = running_loss / (i + 1)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')

            pbar.set_description(f'Training epoch {epoch + 1}')

        
        if logger is not None:
            logger.log({'loss': avg_loss}, step=epoch)
            
        mean, std, success_rate, frames = run_policy(net, env, device=device, mean_inputs=mean_inputs, std_inputs=std_inputs)
        if logger is not None:
            logger.log({'mean_return': mean, 'std_return': std, 'success_rate': success_rate}, step=epoch)
            logger.log({
                'video/video': wandb.Video(frames, fps=30, format='mp4')
            }, step=epoch)
        


def main(_):

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    
    run_name = f"{FLAGS.env}_bc_seed_{FLAGS.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(entity=FLAGS.entity, project='adroit-bc', name=run_name, config=FLAGS)
      
    dataset = minari.load_dataset(FLAGS.env, download=True)
    env = dataset.recover_environment()

    N = FLAGS.ndemos
    sample = dataset.sample_episodes(n_episodes=1)
    T = sample[0].observations.shape[0] - 1
    obs_dim = sample[0].observations.shape[-1]
    print("N", N)
    print("T", T)
    print("obs_dim", obs_dim)
    
    action_dim = sample[0].actions.shape[-1]
    print("action_dim", action_dim)

    inputs = torch.zeros((N, T, obs_dim))
    targets = torch.zeros((N, T, action_dim))

    dataset.set_seed(FLAGS.seed)
    episodes = dataset.sample_episodes(n_episodes=N)
    for i, episode in enumerate(episodes):
        inputs[i] = torch.from_numpy(episode.observations[:-1])
        targets[i] = torch.from_numpy(episode.actions)

    print("inputs", inputs.shape)

    new_inputs = inputs.reshape(N * T, obs_dim)  # [N * T, obs_dim]
    targets = targets.reshape(N * T, action_dim)  # [N * T, action_dim]

    # normalize inputs
    mean_inputs = new_inputs.mean(dim=0)
    std_inputs = new_inputs.std(dim=0)
    
    new_inputs = (new_inputs - mean_inputs) / (std_inputs + 1e-8)

    train_dataset = torch.utils.data.TensorDataset(new_inputs, targets)
    print("train_dataset", len(train_dataset))

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
    nbodies = 25
    
    is_mixed = False if FLAGS.is_mixed == "False" else True
    use_positional_encoding = False if FLAGS.use_positional_encoding == "False" else True
    if FLAGS.network_type == 'mlp':
        net = MLP(embedding_dim * nbodies, hidden_sizes=(dim_feedforward,dim_feedforward))
    elif FLAGS.network_type == 'transformer':
        embedding_dim *= nheads
        net = Transformer(nbodies, embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, nlayers=FLAGS.nlayers, use_positional_encoding=use_positional_encoding)
    elif FLAGS.network_type == 'soft_bias_transformer':
        embedding_dim *= nheads
        net = SoftBiasTransformer(nbodies, FLAGS.env, embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, nlayers=FLAGS.nlayers)
    elif FLAGS.network_type == 'body_transformer':
        embedding_dim *= nheads
        net = BodyTransformer(nbodies, FLAGS.env, embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, num_layers=FLAGS.nlayers, is_mixed=is_mixed, use_positional_encoding=use_positional_encoding, first_hard_layer=FLAGS.first_hard_layer)

    model = BodyNet(FLAGS.env, net, action_dim=action_dim, embedding_dim=embedding_dim, global_input=FLAGS.network_type=='mlp', device=device)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("training on", device)

    model.to(device)

    mean_inputs = mean_inputs.to(device)
    std_inputs = std_inputs.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr)
    criterion = torch.nn.MSELoss()
    
    train(model, train_dataloader, optimizer, criterion, env=env, device=device, logger=wandb, mean_inputs=mean_inputs, std_inputs=std_inputs)

    print("Done")


if __name__ == '__main__':
    app.run(main)