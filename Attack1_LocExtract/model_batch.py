import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from functools import lru_cache

class GraphConvolution(nn.Module):
    def __init__(self, A, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()
        self.A = A

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.linear.weight.size(0))
        self.linear.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        adj = self.A
        support = self.linear(input)
        output = torch.matmul(adj, support)
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

class GCN(nn.Module):
    def __init__(self, A, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(A, channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x)

        return x

class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat


class CombinedModel(nn.Module):
    def __init__(
        self,
        args,
        in_features: int,
        num_users: int,
        num_cats: int,
        num_pois: int,
        poi_idx2cat_idx_dict,
        user_id2idx_dict,
        A,
    ) -> None:
        super(CombinedModel, self).__init__()
        self.poi_idx2cat_idx_dict = poi_idx2cat_idx_dict
        self.user_id2idx_dict = user_id2idx_dict

        self.poi_embed_model = GCN(
            A = A,
            ninput=args.gcn_nfeat,
            nhid=args.gcn_nhid,
            noutput=args.poi_embed_dim,
            dropout=args.gcn_dropout,
        )

        # %% Model2: User embedding model, nn.embedding
        self.user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)

        # %% Model3: Time Model
        self.time_embed_model = Time2Vec("sin", out_dim=args.time_embed_dim)

        # %% Model4: Category embedding model
        self.cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)

        # %% Model5: Sequence model
        self.seq_input_embed = (
            args.poi_embed_dim
            + args.user_embed_dim
            + args.time_embed_dim
            + args.cat_embed_dim
        )
        self.seq_model = TransformerModel(
            num_pois,
            num_cats,
            self.seq_input_embed,
            args.transformer_nhead,
            args.transformer_nhid,
            args.transformer_nlayers,
            dropout=args.transformer_dropout,
        )

        self._device = args.device

    def _input_traj_to_embeddings(self, sample, poi_embeddings):
        # Parse sample
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [0 for each in input_seq]

        # User to embedding
        user_id = traj_id.split("_")[0]
        user_idx = self.user_id2idx_dict[user_id]
        input = torch.LongTensor([user_idx]).to(device=self._device)
        user_embedding = self.user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)

        # POI to embedding and fuse embeddings
        input_seq_embed = []
        for idx in range(len(input_seq)):
            poi_embedding = poi_embeddings[input_seq[idx]]
            poi_embedding = torch.squeeze(poi_embedding).to(device=self._device)

            # Time to vector
            time_embedding = self.time_embed_model(
                torch.tensor([input_seq_time[idx]], dtype=torch.float).to(
                    device=self._device
                )
            )
            time_embedding = torch.squeeze(time_embedding).to(device=self._device)

            # Categroy to embedding
            cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=self._device)
            cat_embedding = self.cat_embed_model(cat_idx)
            cat_embedding = torch.squeeze(cat_embedding)

 
            
            fused_embedding1 = torch.cat((user_embedding, poi_embedding), 0)
            fused_embedding2 = torch.cat((time_embedding, cat_embedding), 0)


            # Concat time, cat after user+poi
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)

            # Save final embed
            input_seq_embed.append(concat_embedding)

        return input_seq_embed

    def _input_trajs_to_embeddings(self, samples, poi_embeddings):
        max_seq_len = -1
        for sample in samples:
            input_seq = [each[0] for each in sample[1]]
            max_seq_len = max(max_seq_len, len(input_seq))
        print(max_seq_len)

        traj_ids = []
        input_seqs = []
        input_seqs_time = []
        input_seqs_cat = []
        user_ids = []
        valid_masks = torch.zeros((len(samples), max_seq_len)).to(device=self._device)

        for idx,sample in enumerate(samples):
            traj_id = sample[0]
            input_seq = [each[0] for each in sample[1]]
            input_seq_time = [each[1] for each in sample[1]]
            input_seq_cat = [0 for each in input_seq]
            user_id = traj_id.split("_")[0]
            user_idx = self.user_id2idx_dict[user_id]
            valid_masks[idx,:len(input_seq)] = 1

            # last pad
            if len(input_seq)<max_seq_len:
                gap = max_seq_len-len(input_seq)
                input_seq += [input_seq[-1]]*gap
                input_seq_time += [input_seq_time[-1]]*gap
                input_seq_cat += [input_seq_cat[-1]]*gap
            assert len(input_seq)==max_seq_len
            assert len(input_seq_time)==max_seq_len
            assert len(input_seq_cat)==max_seq_len

            traj_ids.append(traj_id)
            input_seqs.append(input_seq)
            input_seqs_time.append(input_seq_time)
            input_seqs_cat.append(input_seq_cat)
            user_ids.append(user_idx)

        # get User embeddings
        input = torch.LongTensor(user_ids).to(device=self._device)
        batch_user_embeddings = self.user_embed_model(input)
        batch_user_embeddings = torch.squeeze(batch_user_embeddings)
        # print(batch_user_embeddings.shape)

        # get Categroy embeddings
        input_seqs_cat = torch.LongTensor(input_seqs_cat).to(device=self._device)
        batch_cat_embeddings = self.cat_embed_model(input_seqs_cat)
        batch_cat_embeddings = torch.squeeze(batch_cat_embeddings)

        # get Poi embeddings
        input_seqs = torch.LongTensor(input_seqs).to(device=self._device)
        batch_poi_embeddings = poi_embeddings[input_seqs]
        batch_poi_embeddings = torch.squeeze(batch_poi_embeddings).to(device=self._device)

        # get Time embeddings
        input_seqs_time = torch.tensor(input_seqs_time, dtype=torch.float).to(device=self._device)
        input_seqs_time = input_seqs_time.unsqueeze(-1)
        batch_time_embeddings = self.time_embed_model(input_seqs_time)
        batch_time_embeddings = torch.squeeze(batch_time_embeddings).to(device=self._device)


        # solve border case
        if len(batch_cat_embeddings.shape)==2: # length = 1
            batch_cat_embeddings = batch_cat_embeddings.unsqueeze(1)
        if len(batch_poi_embeddings.shape)==2: # length = 1
            batch_poi_embeddings = batch_poi_embeddings.unsqueeze(1)
        if len(batch_time_embeddings.shape)==2: # length = 1
            batch_time_embeddings = batch_time_embeddings.unsqueeze(1)

        batch_user_embeddings = batch_user_embeddings.unsqueeze(1).repeat(1,batch_poi_embeddings.shape[1],1)
        fused_embedding1 = torch.cat((batch_user_embeddings, batch_poi_embeddings), -1)
        fused_embedding2 = torch.cat((batch_time_embeddings, batch_cat_embeddings), -1)
        concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)
        return concat_embedding

    def _src_mask(self, batch_size: int):
        return self.seq_model.generate_square_subsequent_mask(batch_size).to(
            self._device
        )

    def forward(self, X: torch.Tensor, batch) -> torch.Tensor:
        # For padding
        batch_input_seqs = []
        batch_seq_lens = []
        batch_seq_embeds = []
        batch_seq_labels_poi = []
        batch_seq_labels_time = []
        batch_seq_labels_cat = []

        poi_embeddings = self.poi_embed_model(X)

        samples = []
        # Convert input seq to embeddings
        for sample in batch:
            # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
            traj_id = sample[0]
            input_seq = [each[0] for each in sample[1]]
            label_seq = [each[0] for each in sample[2]]
            input_seq_time = [each[1] for each in sample[1]]
            label_seq_time = [each[1] for each in sample[2]]
            label_seq_cats = [0 for each in label_seq]
            samples.append(sample)
            batch_seq_lens.append(len(input_seq))
            batch_input_seqs.append(input_seq)
            batch_seq_labels_poi.append(torch.LongTensor(label_seq))
            batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
            batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

        batch_seq_embeds = self._input_trajs_to_embeddings(samples, poi_embeddings)
        
        # Pad seqs for batch training
        batch_padded = pad_sequence(
            batch_seq_embeds, batch_first=True, padding_value=-1
        )
        label_padded_poi = pad_sequence(
            batch_seq_labels_poi, batch_first=True, padding_value=-1
        )
        label_padded_time = pad_sequence(
            batch_seq_labels_time, batch_first=True, padding_value=-1
        )
        label_padded_cat = pad_sequence(
            batch_seq_labels_cat, batch_first=True, padding_value=-1
        )

        # Feedforward
        x = batch_padded.to(device=self._device, dtype=torch.float)
        y = {}
        y["poi"] = label_padded_poi.to(device=self._device, dtype=torch.long)
        y["time"] = label_padded_time.to(device=self._device, dtype=torch.float)
        y["cat"] = label_padded_cat.to(device=self._device, dtype=torch.long)
        src_mask = self._src_mask(x.shape[1])
        y["pred_poi"], y["pred_time"], y["pred_cat"] = self.seq_model(x, src_mask)
        y["pred_poi_adjusted"] = y["pred_poi"]

        batch_results = {}
        batch_results["input_seqs"] = batch_input_seqs
        batch_results["seq_lens"] = batch_seq_lens
        batch_results["seq_embeds"] = batch_seq_embeds
        batch_results["seq_labels_poi"] = batch_seq_labels_poi
        batch_results["seq_labels_time"] = batch_seq_labels_time
        batch_results["seq_labels_cat"] = batch_seq_labels_cat
        return y, batch_results