from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
from rl_games.algos_torch.d2rl import D2RLNet
from rl_games.algos_torch.sac_helper import  SquashedNormal
from rl_games.common.layers.recurrent import  GRUWithDones, LSTMWithDones
from rl_games.common.layers.value import  TwoHotEncodedValue, DefaultValue
from rl_games.algos_torch.layers import symexp, symlog

from rl_games.algos_torch.network_builder import NetworkBuilder

from rl_games.algos_torch.structure_transformers import (
    ObsTokenizer, ActionDetokenizer, ValueDetokenizer,
    MLP, Transformer, BodyTransformer, BIBActorCritic, BodyLevelActor, BodyLevelCritic,
)


class TransformerBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self, **kwargs)
        self.kwargs = kwargs
        
    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            self.actions_num = kwargs.pop('actions_num')
            self.value_size = kwargs.pop('value_size', 1)
            self.mapping = kwargs.pop('mapping', None)

            self.centrality_degree = kwargs.pop('centrality_degree', None)
            self.shortest_path_matrix = kwargs.pop('shortest_path_matrix', None)
            self.adjacency_matrix = kwargs.pop('adjacency_matrix', None)

            
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            device = 'cuda'
            obs_dim = sum([len(v[0]) for v in self.mapping['map'].values()])

            # actor
            if self.network_name in ['transformer_shared', 'body_transformer_shared']:
                shared_tokenizer = ObsTokenizer(self.mapping, output_dim=self.dim_embeddings, stack_time=False).to(device)
                shared_trunk = self._build_transformer(self.network_name.replace('_shared', '')).to(device)
                actor_head = ActionDetokenizer(self.mapping, shared_trunk.output_dim, action_dim=self.actions_num, use_mlp=True)
                critic_head = ValueDetokenizer(self.mapping, shared_trunk.output_dim, use_mlp=True)

                actor = nn.Sequential(
                    shared_tokenizer,
                    shared_trunk,
                    actor_head,
                )
                critic = nn.Sequential(
                    shared_tokenizer,
                    shared_trunk,
                    critic_head,
                )
                # dummy_obs = {'obs': torch.zeros((128, 100)).to(device)}
            else:
                actor_tokenizer = ObsTokenizer(self.mapping, output_dim=self.dim_embeddings, stack_time=False, shared=self.shared_tokenizer).to(device)
                actor_trunk = self._build_transformer(self.network_name).to(device)
                actor_detokenizer = ActionDetokenizer(self.mapping, actor_trunk.output_dim, action_dim=self.actions_num, global_input=self.network_name == 'mlp', shared=self.shared_tokenizer).to(device)
                actor = BodyLevelActor(actor_tokenizer, actor_trunk, actor_detokenizer).to(device)

                critic_type = self.critic_type if self.critic_type != 'same' else self.network_name
                critic_trunk = self._build_transformer(critic_type).to(device)
                critic_tokenizer = ObsTokenizer(self.mapping, output_dim=self.dim_embeddings, stack_time=False, shared=self.shared_tokenizer).to(device)
                critic_detokenizer = ValueDetokenizer(self.mapping, critic_trunk.output_dim, global_input=critic_type == 'mlp', shared=self.shared_tokenizer).to(device)
                critic = BodyLevelCritic(critic_tokenizer, critic_trunk, critic_detokenizer).to(device)

            self.actor_critic = BIBActorCritic(actor, critic, self.actions_num, self.fixed_sigma).to(device)


        def forward(self, obs_dict):
            mu, logstd, value = self.actor_critic(obs_dict)
            return mu, logstd, value, None

        def _build_transformer(self, network_type): # Without tokenizer and detokenizer
            if network_type == 'transformer':
                return Transformer(self.dim_embeddings, self.dim_feedforward, self.num_heads, self.num_layers)
            elif network_type == 'body_transformer':
                return BodyTransformer(self.mapping, self.dim_embeddings, self.dim_feedforward, self.num_heads, self.num_layers, bias_type=self.bias_type)
            elif network_type == 'mlp':
                return MLP(self.dim_embeddings*self.nbodies, (self.dim_feedforward, self.dim_feedforward))
            else:
                raise NotImplementedError("Network type {} not implemented".format(network_type))

        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return False
        
        def is_stack(self):
            return self.network_name == 'hard_structure_transformer'

        def get_default_rnn_state(self):
            return None              

        def load(self, params):
            self.separate = params.get('separate', False)
            self.lookback_steps = params.get('lookback_steps', 1)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = False
            self.has_space = 'space' in params
            self.central_value = False
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            self.space_config = params['space']['continuous']
            self.fixed_sigma = self.space_config['fixed_sigma']
            
            self.has_cnn = False

            self.network_name = params.get('name', 'transformer')

            # TODO: Tune these parameters
            self.dim_embeddings = params['transformer']['dim_embeddings']
            self.num_heads = params['transformer']['num_heads']
            self.dim_feedforward = params['transformer']['dim_feedforward']
            self.num_layers = params['transformer']['num_layers']
            self.bias_type = params['transformer'].get('bias_type', 'mixed')
            self.critic_type = params['transformer'].get('critic_type', 'same')
            self.shared_tokenizer = params['transformer'].get('shared_tokenizer', False)

            self.nbodies = len(self.mapping['map'].keys())



    def build(self, name, **kwargs):
        kwargs = {**kwargs, **self.kwargs} # merge kwargs (with priority to self.kwargs)
        net = TransformerBuilder.Network(self.params, **kwargs)
        return net


class SoftStructureTransformerBuilder(TransformerBuilder):

    def __init__(self, **kwargs):
        centrality_degree = torch.Tensor([4, 1, 2, 3, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1])
        shortest_path_matrix = torch.Tensor([[0, 1, 1, 2, 3, 4, 5, 3, 4, 5, 1, 2, 3, 1, 2, 3],
                                    [1, 0, 2, 3, 4, 5, 6, 4, 5, 6, 2, 3, 4, 2, 3, 4],
                                    [1, 2, 0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4],
                                    [2, 3, 1, 0, 1, 2, 3, 1, 2, 3, 3, 4, 5, 3, 4, 5],
                                    [3, 4, 2, 1, 0, 1, 2, 2, 3, 4, 4, 5, 6, 4, 5, 6],
                                    [4, 5, 3, 2, 1, 0, 1, 3, 4, 5, 5, 6, 7, 5, 6, 7],
                                    [5, 6, 4, 3, 2, 1, 0, 4, 5, 6, 6, 7, 8, 6, 7, 8],
                                    [3, 4, 2, 1, 2, 3, 4, 0, 1, 2, 4, 5, 6, 4, 5, 6],
                                    [4, 5, 3, 2, 3, 4, 5, 1, 0, 1, 5, 6, 7, 5, 6, 7],
                                    [5, 6, 4, 3, 4, 5, 6, 2, 1, 0, 6, 7, 8, 6, 7, 8],
                                    [1, 2, 2, 3, 4, 5, 6, 4, 5, 6, 0, 1, 2, 2, 3, 4],
                                    [2, 3, 3, 4, 5, 6, 7, 5, 6, 7, 1, 0, 1, 3, 4, 5],
                                    [3, 4, 4, 5, 6, 7, 8, 6, 7, 8, 2, 1, 0, 4, 5, 6],
                                    [1, 2, 2, 3, 4, 5, 6, 4, 5, 6, 2, 3, 4, 0, 1, 2],
                                    [2, 3, 3, 4, 5, 6, 7, 5, 6, 7, 3, 4, 5, 1, 0, 1],
                                    [3, 4, 4, 5, 6, 7, 8, 6, 7, 8, 4, 5, 6, 2, 1, 0]])
        kwargs['centrality_degree'] = centrality_degree
        kwargs['shortest_path_matrix'] = shortest_path_matrix
        super().__init__(**kwargs)
    
class HardStructureTransformerBuilder(TransformerBuilder):

    def __init__(self, **kwargs):
        centrality_degree = torch.Tensor([4, 1, 2, 3, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1])
        shortest_path_matrix = torch.Tensor([[0, 1, 1, 2, 3, 4, 5, 3, 4, 5, 1, 2, 3, 1, 2, 3],
                                    [1, 0, 2, 3, 4, 5, 6, 4, 5, 6, 2, 3, 4, 2, 3, 4],
                                    [1, 2, 0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4],
                                    [2, 3, 1, 0, 1, 2, 3, 1, 2, 3, 3, 4, 5, 3, 4, 5],
                                    [3, 4, 2, 1, 0, 1, 2, 2, 3, 4, 4, 5, 6, 4, 5, 6],
                                    [4, 5, 3, 2, 1, 0, 1, 3, 4, 5, 5, 6, 7, 5, 6, 7],
                                    [5, 6, 4, 3, 2, 1, 0, 4, 5, 6, 6, 7, 8, 6, 7, 8],
                                    [3, 4, 2, 1, 2, 3, 4, 0, 1, 2, 4, 5, 6, 4, 5, 6],
                                    [4, 5, 3, 2, 3, 4, 5, 1, 0, 1, 5, 6, 7, 5, 6, 7],
                                    [5, 6, 4, 3, 4, 5, 6, 2, 1, 0, 6, 7, 8, 6, 7, 8],
                                    [1, 2, 2, 3, 4, 5, 6, 4, 5, 6, 0, 1, 2, 2, 3, 4],
                                    [2, 3, 3, 4, 5, 6, 7, 5, 6, 7, 1, 0, 1, 3, 4, 5],
                                    [3, 4, 4, 5, 6, 7, 8, 6, 7, 8, 2, 1, 0, 4, 5, 6],
                                    [1, 2, 2, 3, 4, 5, 6, 4, 5, 6, 2, 3, 4, 0, 1, 2],
                                    [2, 3, 3, 4, 5, 6, 7, 5, 6, 7, 3, 4, 5, 1, 0, 1],
                                    [3, 4, 4, 5, 6, 7, 8, 6, 7, 8, 4, 5, 6, 2, 1, 0]])
        adjacency_matrix = shortest_path_matrix < 2  
        kwargs['centrality_degree'] = centrality_degree
        kwargs['adjacency_matrix'] = adjacency_matrix
        super().__init__(**kwargs)
        
    