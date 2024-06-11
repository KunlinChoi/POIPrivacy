import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence



class RNN_POI(nn.Module):
    def __init__(self, hidden_size, loc_size,loc_emb_size,tim_emb_size,usr_size,usr_emb_size):
        super(RNN_POI, self).__init__()
        # embedding parameters
        self.loc_size = loc_size
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.usr_size = usr_size
        self.usr_emb_size = usr_emb_size

        input_size = self.loc_emb_size + self.tim_emb_size + tim_emb_size
        # Embedding
        self.emb_loc = nn.Embedding(
            self.loc_size, self.loc_emb_size,
            padding_idx=0)
        
        self.tim_emb_size = tim_emb_size

        self.time_emb=  nn.Embedding(num_embeddings=1000, embedding_dim=128,padding_idx=0)

        self.emb_usr = nn.Embedding(
            self.usr_size, self.usr_emb_size,
            padding_idx=0)
        
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, 3, batch_first=True)

        self.init_weights()
        self.fc = nn.Linear(hidden_size, self.loc_size)
        self.dropout = nn.Dropout(0.2)

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for
        consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters()
              if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters()
              if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters()
             if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, user, loc,tim,origin_len):
        loc_emb = self.emb_loc(loc)
        tim = tim.view(-1, 1)
        tim = tim * 1000
        tim = tim.long()
        tim_emb = self.time_emb(tim)
        tim_emb = tim_emb.view(loc_emb.size(0),loc_emb.size(1),self.tim_emb_size)
        user_emb = self.emb_usr(user)
        x = torch.cat((user_emb, loc_emb), 2)
        x = torch.cat((x, tim_emb), 2)
        h0 = torch.zeros(3, x.size(0), self.hidden_size).cuda()
        out, h0 = self.rnn(x, h0)
        ret = self.fc(out)
        return ret
