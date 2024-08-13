import torch
import torch.nn as nn

from .masked_transformer_encoder import MaskedTransformerEncoder

from .mappings import Mapping
    
class Tokenizer(torch.nn.Module):
    def __init__(self, env_name, output_dim, output_activation=None, device='cuda'):
        super(Tokenizer, self).__init__()

        self.mapping = Mapping(env_name)
        self.map = self.mapping.get_map()

        self.output_dim = output_dim
        self.output_activation = output_activation

        self.device = device

        self.zero_token = torch.nn.Parameter(torch.zeros(1, output_dim), requires_grad=False)

        base = lambda input_dim : torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))
        self.tokenizers = torch.nn.ModuleDict()
        for k in self.map.keys():
            self.tokenizers[k] = base(len(self.map[k][0])) 
            if output_activation is not None:
                self.tokenizers[k] = torch.nn.Sequential(self.tokenizers[k], output_activation)

    def forward(self, x):
        x = self.mapping.create_observation(x)
        outputs = []
        for key in x.keys():
            inputs = x[key].to(self.device)
            if inputs.shape[-1] == 0:
                outputs.append(self.zero_token.expand(*inputs.shape[:-1], -1).unsqueeze(1))
            else:
                outputs.append(self.tokenizers[key](inputs).unsqueeze(1))
        return torch.cat(outputs, dim=1)
            
class Detokenizer(torch.nn.Module):
    def __init__(self, env_name, embedding_dim, action_dim, num_layers=1, global_input=False, output_activation=None, device='cuda'):
        super(Detokenizer, self).__init__()

        self.mapping = Mapping(env_name)
        self.map = self.mapping.get_map()

        self.nbodies = len(self.map.keys())

        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.output_activation = output_activation

        self.device = device

        base = lambda output_dim : torch.nn.Linear(embedding_dim, output_dim)
        self.detokenizers = torch.nn.ModuleDict()
        if global_input:
            self.detokenizers['global'] = base(action_dim)
            if output_activation is not None:
                self.detokenizers['global'] = torch.nn.Sequential(self.detokenizers['global'], output_activation)
        else:
            for k in self.map.keys():
                self.detokenizers[k] = base(len(self.map[k][1])) 
                if output_activation is not None:
                    self.detokenizers[k] = torch.nn.Sequential(self.detokenizers[k], output_activation)

    def forward(self, x, weights=None):

        if 'global' in self.detokenizers:
            return self.detokenizers['global'](x.to(self.device))
        
        action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
        for i, k in enumerate(self.map.keys()):
            curr_action = self.detokenizers[k](x[:,i,:])
            action[:, self.map[k][1]] = curr_action
        return action
    
    def weighted_sum(self, x, weights):
        return torch.sum(x * weights.unsqueeze(-1), dim=1) # (B, action_dim)
    
    def weighted_sum_per_time(self, x, weights):
        return torch.sum(x * weights.unsqueeze(-1), dim=1) # (B, action_dim)
        
    
class Transformer(torch.nn.Module):
    def __init__(self, nbodies, input_dim, dim_feedforward=256, nhead=6, nlayers=3, use_positional_encoding=False):
        super(Transformer, self).__init__()
        encoder_layer =  torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer,num_layers=nlayers)

        self.output_dim = input_dim
        self.use_positional_encoding = use_positional_encoding

        if use_positional_encoding:
            print("Using positional encoding")
            self.embed_absolute_position = nn.Embedding(nbodies, embedding_dim=input_dim) 

    def forward(self, x):
        if self.use_positional_encoding:
            _, nbodies, _ = x.shape
            limb_indices = torch.arange(0, nbodies, device=x.device)
            limb_idx_embedding = self.embed_absolute_position(limb_indices)
            x = x + limb_idx_embedding
        x = self.encoder(x)
        return x
    
class SoftBiasTransformer(Transformer):
    def __init__(self, nbodies, env_name, input_dim, dim_feedforward=256, nhead=6, nlayers=3):
        super(SoftBiasTransformer, self).__init__(nbodies, input_dim, dim_feedforward, nhead, nlayers)
        shortest_path_matrix = Mapping(env_name).shortest_path_matrix
        self.sp_embeddings = torch.nn.Embedding(int(shortest_path_matrix.max())+1, 1, dtype=torch.float)

        self.register_buffer('shortest_path_matrix', shortest_path_matrix.long())

    def forward(self, x):
        sp_embeddings = self.sp_embeddings(self.shortest_path_matrix) # (nbodies x nbodies) -> (nbodies x nbodies x 1)
        sp_embeddings = sp_embeddings.squeeze(-1) # (nbodies x nbodies x 1) -> (nbodies x nbodies)

        x = self.encoder(x, mask=sp_embeddings)

        return x
    
class BodyTransformer(Transformer):
    def __init__(self, nbodies, env_name, input_dim, dim_feedforward=256, nhead=6, num_layers=3, is_mixed=True, use_positional_encoding=False, first_hard_layer=1, random_mask=False):
        super(BodyTransformer, self).__init__(nbodies, input_dim, dim_feedforward, nhead, use_positional_encoding=use_positional_encoding)
        shortest_path_matrix = Mapping(env_name).shortest_path_matrix
        adjacency_matrix = shortest_path_matrix < 2

        if random_mask:
            num_nonzero = torch.sum(adjacency_matrix) - adjacency_matrix.shape[0]
            prob_nonzero = num_nonzero / (adjacency_matrix.shape[0] * adjacency_matrix.shape[0])
            adjacency_matrix = torch.rand(adjacency_matrix.shape) > prob_nonzero
            adjacency_matrix.fill_diagonal_(True)

        self.nbodies = adjacency_matrix.shape[0]

        # We assume (B x nbodies x input_dim) batches
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)

        self.encoder = MaskedTransformerEncoder(encoder_layer, num_layers=num_layers)

        self.is_mixed = is_mixed
        self.use_positional_encoding = use_positional_encoding

        self.first_hard_layer = first_hard_layer

        self.register_buffer('adjacency_matrix', adjacency_matrix)

    def forward(self, x):
        if self.use_positional_encoding:
            limb_indices = torch.arange(0, self.nbodies, device=x.device)
            limb_idx_embedding = self.embed_absolute_position(limb_indices)
            x = x + limb_idx_embedding

        x = self.encoder(x, mask=~self.adjacency_matrix, is_mixed=self.is_mixed, return_intermediate=False, first_hard_layer=self.first_hard_layer)

        return x
        

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
    
class BodyNet(torch.nn.Module):
    def __init__(self, env_name, net, action_dim, embedding_dim, output_activation=None, global_input=False, fixed_std=0.1, device='cuda'):
        super(BodyNet, self).__init__()

        self.std = fixed_std

        self.tokenizer = Tokenizer(env_name, embedding_dim, output_activation=output_activation, device=device)
        self.net = net
        self.detokenizer = Detokenizer(env_name, net.output_dim, action_dim, device=device, global_input=global_input)

        self.tokenizer.to(device)
        self.net.to(device)

    def forward(self, x):
        weights = None 

        x = self.tokenizer(x) # (B, nbodies, embedding_dim)
        x = self.net(x)
        
        x = self.detokenizer(x, weights)
        
        return x

    def mode(self, x):
        return self.forward(x)
    
    def log_prob(self, x, action):

        mu = self.forward(x)
        std = self.std
        return torch.distributions.Normal(mu, std).log_prob(action).sum(1)
    