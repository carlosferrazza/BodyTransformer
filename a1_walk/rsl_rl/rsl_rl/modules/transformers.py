import torch

import torch.nn as nn

from positional_encodings.torch_encodings import PositionalEncoding1D

from .transformer_utils import MixedTransformerEncoder

from rsl_rl.utils.mappings import Mapping

class ObsTokenizer(torch.nn.Module):
    def __init__(self, name, output_dim, stack_time=True, device='cuda'):
        super(ObsTokenizer, self).__init__()

        self.mapping = Mapping(name)
        self.map = self.mapping.get_map()

        self.output_dim = output_dim
        
        self.stack_time = stack_time

        self.device = device

        self.zero_token = torch.nn.Parameter(torch.zeros(1, output_dim), requires_grad=False)

        base = lambda input_dim : torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))
        self.tokenizers = torch.nn.ModuleDict()
        for k in self.map.keys():
            self.tokenizers[k] = base(len(self.map[k][0])) 
            
    def forward(self, x):
        x = self.mapping.create_observation(x, stack_time=self.stack_time)
        outputs = []
        for key in x.keys():
            inputs = x[key].to(self.device)
            if inputs.shape[-1] == 0:
                outputs.append(self.zero_token.expand(*inputs.shape[:-1], -1).unsqueeze(1))
            else:
                outputs.append(self.tokenizers[key](inputs).unsqueeze(1))
        return torch.cat(outputs, dim=1)
    
class ActionTokenizer(torch.nn.Module):
    def __init__(self, name, output_dim, action_dim, device='cuda'):
        super(ActionTokenizer, self).__init__()

        self.mapping = Mapping(name)
        self.map = self.mapping.get_map()

        self.output_dim = output_dim
        self.action_dim = action_dim

        self.device = device

        self.zero_token = torch.nn.Parameter(torch.zeros(1, output_dim), requires_grad=False)

        base = lambda input_dim : torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))
        self.tokenizers = torch.nn.ModuleDict()
        for k in self.map.keys():
            self.tokenizers[k] = base(len(self.map[k][1]))

    def forward(self, x):
        x = self.mapping.create_action(x)
        outputs = []
        for key in x.keys():
            inputs = x[key].to(self.device)
            if inputs.shape[-1] == 0:
                outputs.append(self.zero_token.expand(*inputs.shape[:-1], -1).unsqueeze(1))
            else:
                outputs.append(self.tokenizers[key](inputs).unsqueeze(1))
        return torch.cat(outputs, dim=1)
    
class ActionDetokenizer(torch.nn.Module):
    def __init__(self, name, embedding_dim, action_dim, global_input=False, device='cuda'):
        super(ActionDetokenizer, self).__init__()

        self.mapping = Mapping(name)
        self.map = self.mapping.get_map()

        self.nbodies = len(self.map.keys())

        self.embedding_dim = embedding_dim
        self.action_dim = action_dim

        self.device = device

        base = lambda output_dim : torch.nn.Linear(embedding_dim, output_dim)
        self.detokenizers = torch.nn.ModuleDict()
        if global_input:
            self.detokenizers['global'] = base(action_dim)
        else:
            for k in self.map.keys():
                self.detokenizers[k] = base(len(self.map[k][1])) 

    def forward(self, x):

        if 'global' in self.detokenizers:
            return self.detokenizers['global'](x.to(self.device))
        
        action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
        for i, k in enumerate(self.map.keys()):
            curr_action = self.detokenizers[k](x[:,i,:])
            action[:, self.map[k][1]] = curr_action.float()
        return action
    
class ValueDetokenizer(torch.nn.Module):
    def __init__(self, name, embedding_dim, global_input=False, device='cuda'):
        super(ValueDetokenizer, self).__init__()

        self.mapping = Mapping(name)
        self.map = self.mapping.get_map()

        self.nbodies = len(self.map.keys())

        self.embedding_dim = embedding_dim

        self.device = device

        base = lambda output_dim : torch.nn.Linear(embedding_dim, output_dim)
        self.detokenizers = torch.nn.ModuleDict()
        if global_input:
            self.detokenizers['global'] = base(1)
        else:
            for k in self.map.keys():
                self.detokenizers[k] = base(1) 

    def forward(self, x):

        if 'global' in self.detokenizers:
            return self.detokenizers['global'](x.to(self.device))
        
        values = torch.zeros(x.shape[0], x.shape[1]).to(self.device)
        for i, k in enumerate(self.map.keys()):
            values[:,i] = self.detokenizers[k](x[:,i,:]).squeeze(-1)
        return torch.mean(values, dim=1, keepdim=True)

class BodyActor(nn.Module):
    def __init__(self, name, net, embedding_dim, action_dim, stack_time=True, global_input=False, mu_activation=None, device='cuda'):
        super(BodyActor, self).__init__()

        self.tokenizer = ObsTokenizer(name, embedding_dim, stack_time, device)
        self.net = net
        self.detokenizer = ActionDetokenizer(name, net.output_dim, action_dim, global_input, device=device)

        self.tokenizer.to(device)
        self.net.to(device)
        self.detokenizer.to(device)

        self.mu_activation = mu_activation

    def forward(self, x):
        x = self.tokenizer(x) # (B, nbodies, lookback_steps, embedding_dim)
        
        x = self.net(x)

        x = self.detokenizer(x)

        if self.mu_activation is not None:
            x = self.mu_activation(x)
        
        return x

    def mode(self, x):
        return self.forward(x)

class BodyCritic(nn.Module):
    def __init__(self, mapping, net, embedding_dim, stack_time=True, global_input=False, device='cuda'):
        super(BodyCritic, self).__init__()

        self.tokenizer = ObsTokenizer(mapping, embedding_dim, stack_time, device)
        self.net = net
        self.detokenizer = ValueDetokenizer(mapping, net.output_dim, global_input, device=device)

        self.tokenizer.to(device)
        self.net.to(device)
        self.detokenizer.to(device)

    def forward(self, x):

        x = self.tokenizer(x) # (B, nbodies, lookback_steps, embedding_dim)
        x = self.net(x)
        
        x = self.detokenizer(x)
        
        return x

    def mode(self, x):
        return self.forward(x)
    
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
        self.init_weights()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_weights(self):
        self.apply(self._init_weights)

class Transformer(torch.nn.Module):
    def __init__(self, input_dim, dim_feedforward=256, nhead=6, nlayers=3):
        super(Transformer, self).__init__()
        encoder_layer =  torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer,num_layers=nlayers)

        max_nbodies = 13  # TODO: not hard-coded
        self.embed_absolute_position = nn.Embedding(max_nbodies, embedding_dim=input_dim)  # max_num_limbs

        self.output_dim = input_dim
        self.init_weights()

    def forward(self, x):
        batch_size, nbodies, embedding_dim = x.shape
        limb_indices = torch.arange(0, nbodies, device=x.device)
        limb_idx_embedding = self.embed_absolute_position(limb_indices)
        x = x + limb_idx_embedding
        x = self.encoder(x)
        return x
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_weights(self):
        self.apply(self._init_weights)
    
class SoftBiasTransformer(Transformer):
    def __init__(self, name, input_dim, dim_feedforward=256, nhead=6, nlayers=3):
        super(SoftBiasTransformer, self).__init__(input_dim, dim_feedforward, nhead, nlayers)
        shortest_path_matrix = Mapping(name).shortest_path_matrix
        self.sp_embeddings = torch.nn.Embedding(int(shortest_path_matrix.max())+1, 1, dtype=torch.float)

        self.register_buffer('shortest_path_matrix', shortest_path_matrix.long())

    def forward(self, x):
        sp_embeddings = self.sp_embeddings(self.shortest_path_matrix) # (nbodies x nbodies) -> (nbodies x nbodies x 1)
        sp_embeddings = sp_embeddings.squeeze(-1) # (nbodies x nbodies x 1) -> (nbodies x nbodies)

        x = self.encoder(x, mask=sp_embeddings)

        return x
    
class BodyTransformer(Transformer):
    def __init__(self, name, input_dim, dim_feedforward=256, nhead=6, num_layers=3, is_mixed=True, first_hard_layer=0):
        super(BodyTransformer, self).__init__(input_dim, dim_feedforward, nhead)
        shortest_path_matrix = Mapping(name).shortest_path_matrix
        adjacency_matrix = shortest_path_matrix < 2

        self.nbodies = adjacency_matrix.shape[0]

        # We assume (B x nbodies x input_dim) batches
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)

        self.encoder = MixedTransformerEncoder(encoder_layer, num_layers=num_layers)

        self.embed_absolute_position = nn.Embedding(self.nbodies, embedding_dim=input_dim)

        self.is_mixed = is_mixed

        self.first_hard_layer = first_hard_layer

        self.register_buffer('adjacency_matrix', adjacency_matrix)

        self.init_weights()

    def forward(self, x):
        limb_indices = torch.arange(0, self.nbodies, device=x.device)
        limb_idx_embedding = self.embed_absolute_position(limb_indices)
        x = x + limb_idx_embedding
        x = self.encoder(x, mask=~self.adjacency_matrix, is_mixed=self.is_mixed, return_intermediate=False, first_hard_layer=self.first_hard_layer)

        return x