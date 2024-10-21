import torch

import torch.nn as nn
import torch.nn.functional as F

from .transformer_utils import MixedTransformerEncoder

class Mapping:
    
    def __init__(self, mapping):
        self.map = mapping['map']
        self.shortest_path_matrix = mapping['sp_matrix']

    def get_map(self):
        return self.map
    
    def get_adjacency_matrix(self):
        return self.shortest_path_matrix < 2
    
    def create_observation(self, observations, stack_time=True):
        obs = observations['obs']
        if len(obs.shape) == 1:
            obs = obs.reshape(1, obs.shape[0])
        # obs = obs.reshape(obs.shape[0], -1, self.dim)
        new_obs = {}
        for k, v in self.map.items():
            input_indices, output_indices = v  # e.g. v: ([13, 14, 15, 34, 35, 36, 64, 65, 66], [3, 4, 5])
            new_obs[k] = obs[:, input_indices]
            if stack_time:
                new_obs[k] = new_obs[k].reshape(obs.shape[0], -1)
        return new_obs

    def create_action(self, action):
        new_action = {}
        for k, v in self.map.items():
            new_action[k] = action[:,v[1]]
        return new_action

class ObsTokenizer(torch.nn.Module):
    def __init__(self, mapping, output_dim, stack_time=True, shared=False, device='cuda'):
        super(ObsTokenizer, self).__init__()

        self.mapping = Mapping(mapping)
        self.map = mapping['map']
        self.output_dim = output_dim
        self.stack_time = stack_time
        self.device = device

        self.shared = shared

        if self.shared:
            self.max_limb_obs_dim = max([len(v[0]) for v in self.map.values()])  # 13
            self.tokenizer = torch.nn.Linear(self.max_limb_obs_dim, output_dim)
        else:
            self.tokenizers = torch.nn.ModuleDict()
            for k, v in self.map.items():
                input_indices, output_indices = v
                input_dim = len(input_indices)
                self.tokenizers[k] = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.mapping.create_observation(x, stack_time=self.stack_time)
        outputs = []
        for key in x.keys():
            inputs = x[key].to(self.device)
            if self.shared:
                batch_size, obs_dim = inputs.shape
                padded_inputs = F.pad(inputs, (0, self.max_limb_obs_dim - obs_dim), "constant", 0)
                outputs.append(self.tokenizer(padded_inputs).unsqueeze(1))
            else:
                outputs.append(self.tokenizers[key](inputs).unsqueeze(1))
        return torch.cat(outputs, dim=1)  # [batch_size, nbodies, embedding_dim]


class ActionDetokenizer(torch.nn.Module):
    def __init__(self, mapping, embedding_dim, action_dim, global_input=False, use_mlp=False, shared=False, device='cuda'):
        super(ActionDetokenizer, self).__init__()

        self.mapping = Mapping(mapping)
        self.map = self.mapping.get_map()
        self.nbodies = len(self.map.keys())
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.device = device
        self.shared = shared

        self.detokenizers = torch.nn.ModuleDict()
        if global_input:
            self.detokenizers['global'] = torch.nn.Linear(embedding_dim, action_dim)
        else:
            if shared:
                self.max_limb_action_dim = max([len(v[1]) for v in self.map.values()])
                self.detokenizer = torch.nn.Linear(embedding_dim, self.max_limb_action_dim)
            else:
                for k, v in self.map.items():
                    input_indices, output_indices = v
                    output_dim = len(output_indices)
                    if use_mlp:
                        self.detokenizers[k] = nn.Sequential(
                            torch.nn.Linear(embedding_dim, 256),
                            nn.ReLU(),
                            torch.nn.Linear(256, 256),
                            nn.ReLU(),
                            torch.nn.Linear(256, output_dim)
                        )
                    else:
                        self.detokenizers[k] = torch.nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        if 'global' in self.detokenizers:
            return self.detokenizers['global'](x.to(self.device))
        
        action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
        for i, k in enumerate(self.map.keys()):
            if self.shared:
                limb_action_dim = len(self.map[k][1])
                curr_action = self.detokenizer(x[:, i, :])[:, :limb_action_dim]
            else:
                curr_action = self.detokenizers[k](x[:,i,:])
            action[:, self.map[k][1]] = curr_action.float()
        return action


class ValueDetokenizer(torch.nn.Module):
    def __init__(self, mapping, embedding_dim, global_input=False, use_mlp=False, shared=False, device='cuda'):
        super(ValueDetokenizer, self).__init__()

        self.mapping = Mapping(mapping)
        self.map = self.mapping.get_map()
        self.nbodies = len(self.map.keys())
        self.embedding_dim = embedding_dim
        self.device = device
        self.shared = shared

        self.detokenizers = torch.nn.ModuleDict()
        if global_input:
            self.detokenizers['global'] = torch.nn.Linear(embedding_dim, 1)
        else:
            if self.shared:
                self.detokenizer = torch.nn.Linear(embedding_dim, 1)
            else:
                for k in self.map.keys():
                    if use_mlp:
                        self.detokenizers[k] = nn.Sequential(
                            torch.nn.Linear(embedding_dim, 256),
                            nn.ReLU(),
                            torch.nn.Linear(256, 256),
                            nn.ReLU(),
                            torch.nn.Linear(256, 1)
                        )
                    else:
                        self.detokenizers[k] = torch.nn.Linear(embedding_dim, 1)

    def forward(self, x):
        if 'global' in self.detokenizers:
            return self.detokenizers['global'](x.to(self.device))

        values = torch.zeros(x.shape[0], x.shape[1]).to(self.device)
        for i, k in enumerate(self.map.keys()):
            if self.shared:
                values[:,i] = self.detokenizer(x[:,i,:]).squeeze(-1)
            else:
                values[:,i] = self.detokenizers[k](x[:,i,:]).squeeze(-1)
        return torch.mean(values, dim=1, keepdim=True)


class BodyLevelActor(nn.Module):
    def __init__(self, tokenizer, trunk, detokenizer):
        super(BodyLevelActor, self).__init__()
        self.tokenizer = tokenizer
        self.trunk = trunk
        self.detokenizer = detokenizer
        self.model = nn.Sequential(tokenizer, trunk, detokenizer)

    def forward(self, x):
        return self.model(x)


class BodyLevelCritic(nn.Module):
    def __init__(self, tokenizer, trunk, detokenizer):
        super(BodyLevelCritic, self).__init__()
        self.tokenizer = tokenizer
        self.trunk = trunk
        self.detokenizer = detokenizer
        self.model = nn.Sequential(tokenizer, trunk, detokenizer)

    def forward(self, x):
        return self.model(x)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_sizes=(1024, 1024), activation=torch.nn.ReLU()):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.layers = []
        in_size = input_dim
        for next_size in hidden_sizes:
            self.layers.append(torch.nn.Linear(in_size, next_size))
            self.layers.append(self.activation)
            in_size = next_size
        self.layers.append(torch.nn.Linear(in_size, input_dim))
        
        self.model = torch.nn.Sequential(*self.layers)

        self.output_dim = input_dim

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)


class Transformer(torch.nn.Module):
    def __init__(self, input_dim, dim_feedforward=256, nhead=6, nlayers=3, positional_embeding_size=16):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)

        self.embed_absolute_position = nn.Embedding(positional_embeding_size, embedding_dim=input_dim)  # max_num_limbs
        self.encoder = torch.nn.TransformerEncoder(encoder_layer,num_layers=nlayers)

        self.output_dim = input_dim

    def forward(self, x):
        batch_size, nbodies, embedding_dim = x.shape
        limb_indices = torch.arange(0, nbodies, device=x.device)
        limb_idx_embedding = self.embed_absolute_position(limb_indices)
        x = x + limb_idx_embedding

        x = self.encoder(x)
        return x


class SoftBiasTransformer(Transformer):
    def __init__(self, mapping, input_dim, dim_feedforward=256, nhead=6, nlayers=3):
        super(SoftBiasTransformer, self).__init__(input_dim, dim_feedforward, nhead, nlayers)
        shortest_path_matrix = Mapping(mapping).shortest_path_matrix
        self.sp_embeddings = torch.nn.Embedding(int(shortest_path_matrix.max())+1, 1, dtype=torch.float)

        self.register_buffer('shortest_path_matrix', shortest_path_matrix.long())

    def forward(self, x):
        sp_embeddings = self.sp_embeddings(self.shortest_path_matrix) # (nbodies x nbodies) -> (nbodies x nbodies x 1)
        sp_embeddings = sp_embeddings.squeeze(-1) # (nbodies x nbodies x 1) -> (nbodies x nbodies)

        x = self.encoder(x, mask=sp_embeddings)

        return x

class BodyTransformer(Transformer):
    def __init__(self, mapping, input_dim, dim_feedforward=256, nhead=6, num_layers=3, bias_type='mixed'):
        super(BodyTransformer, self).__init__(input_dim, dim_feedforward, nhead)
        shortest_path_matrix = Mapping(mapping).shortest_path_matrix
        adjacency_matrix = shortest_path_matrix < 2

        self.nbodies = adjacency_matrix.shape[0]

        # We assume (B x nbodies x input_dim) batches
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)

        self.bias_type = bias_type
        self.embed_absolute_position = nn.Embedding(self.nbodies, embedding_dim=input_dim)
        self.encoder = MixedTransformerEncoder(encoder_layer, num_layers=num_layers, bias_type=bias_type, nbodies=self.nbodies)

        self.register_buffer('adjacency_matrix', adjacency_matrix)

    def forward(self, x):
        limb_indices = torch.arange(0, self.nbodies, device=x.device)
        limb_idx_embedding = self.embed_absolute_position(limb_indices)
        x = x + limb_idx_embedding
        x = self.encoder(x, mask=~self.adjacency_matrix, return_intermediate=False)

        return x

class BIBActorCritic(nn.Module):

    def __init__(self, actor, critic, actions_num, fixed_sigma=True):
        super().__init__()

        self.actor = actor
        self.critic = critic

        print("actor class {}".format(actor.__class__.__name__))
        print("critic class {}".format(critic.__class__.__name__))

        actor_nparams = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        critic_nparams = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        print('actor params:', actor_nparams)
        print('critic params:', critic_nparams)
        # print('total params:', actor_nparams + value_nparams)

        self.actions_num = actions_num
        self.fixed_sigma = fixed_sigma

        if self.fixed_sigma:
            self.logstd = nn.Parameter(torch.ones(self.actions_num, requires_grad=True, dtype=torch.float32) * 0.0, requires_grad=True)
        else:
            self.logstd = torch.nn.Linear(self.dim_embeddings, self.actions_num)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('total params:', total_params)
        self.init_weights()

    def forward(self, observations):

        a_out = c_out = observations
    
        mu = self.actor(a_out) 
        value = self.critic(c_out)
       
        if self.fixed_sigma:
            logstd = self.logstd
        else:
            logstd = self.logstd(a_out)

        return mu, logstd, value

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_weights(self):
        self.apply(self._init_weights)
