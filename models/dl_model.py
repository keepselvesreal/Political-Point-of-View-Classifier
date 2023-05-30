# reference: rnn attention -> https://github.com/gucci-j/pytorch-imdb-cv/blob/master/src/model.py, cnn -> https://github.com/kh-kim/simple-ntc/blob/master/simple_ntc/models/cnn.py
    
import math
import torch
from torch import nn
from torch.nn import functional as F

class Rnn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        rnn_cell = getattr(nn, config.rnn_type)
        self.model = rnn_cell(config.embedding_dim, config.hidden_dim,
                           num_layers=config.num_layers,
                           bidirectional=config.bidirectional,
                           dropout=config.dropout)
        self.bi_fc = nn.Linear(2 * config.hidden_dim, config.output_dim)
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.query_dim = 2 * config.hidden_dim if config.bidirectional else config.hidden_dim
        self.scale = 1. / math.sqrt(self.query_dim)

    def attention(self, query, key, value):
        # query == hidden: (batch_size, hidden_dim * 2)
        # key/value == gru_output: (sentence_length, batch_size, hidden_dim * 2)
        query = query.unsqueeze(1) # (batch_size, 1, hidden_dim * 2)
        key = key.transpose(0, 1).transpose(1, 2) # (batch_size, hidden_dim * 2, sentence_length)

        # bmm: batch matrix-matrix multiplication
        attention_weight = torch.bmm(query, key) # (batch_size, 1, sentence_length)
        attention_weight = F.softmax(attention_weight.mul_(self.scale), dim=2) # normalize sentence_length's dimension

        value = value.transpose(0, 1) # (batch_size, sentence_length, hidden_dim * 2)
        attention_output = torch.bmm(attention_weight, value) # (batch_size, 1, hidden_dim * 2)
        attention_output = attention_output.squeeze(1) # (batch_size, hidden_dim * 2)

        return attention_output, attention_weight.squeeze(1)

    def forward(self, data_dict, stage='train'):
        if self.config.seq_type == 'packing':
            inputs, lengths = data_dict['input_ids'], data_dict['length']
        else:
            inputs = data_dict["input_ids"]
        # inputs: (sentence_length, batch_size)
        inputs = inputs.transpose(0, 1)
        embed = self.embedding(inputs)
        embed = self.dropout(embed)
        # embedded: (sentence_length, batch_size, embedding_dim)
        if self.config.seq_type == 'packing':
            embed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu().numpy(), enforce_sorted=False) # 위에서 seq_first로 바꿨단 걸 고려못했네! batch_first=True 제외시켜야
        
        # gru_output: (sentence_length, batch_size, hidden_dim * 2)
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        gru_output, hidden = self.model(embed)
        if self.config.seq_type == 'packing':
            gru_output, unpacked_lengths = nn.utils.rnn.pad_packed_sequence(gru_output) # packing data 입력 시 출력 결과 주의해야
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        
        if self.config.bidirectional:
            # concat the final output of forward direction and backward direction
            # hidden: (batch_size, hidden_dim * 2)
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        attention_output, attention_weight = self.attention(query=hidden, key=gru_output, value=gru_output)
        
        if self.config.bidirectional:
          output = self.bi_fc(attention_output)
        else:
          output = self.fc(attention_output)
        output = F.softmax(output, dim=1)
        output_dict = {'output': output}
        return output_dict
    

class Cnn(nn.Module):
    def __init__(self, config):
      self.use_batch_norm = config.use_batch_norm
      self.vocab_size = config.vocab_size
      self.embedding_dim = config.embedding_dim
      self.output_dim = config.output_dim
      self.use_batch_norm = config.use_batch_norm
      self.dropout = config.dropout
      self.window_size_list = config.window_size_list
      self.num_filter_list = config.num_filter_list

      super().__init__()
      self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
      self.feature_extractors = nn.ModuleList()
      for window_size, num_filter in zip(self.window_size_list, self.num_filter_list):
          self.feature_extractors.append(
              nn.Sequential(
                  nn.Conv2d(
                      in_channels=1, 
                      out_channels=num_filter,
                      kernel_size=(window_size, self.embedding_dim),
                  ),
                  nn.ReLU(),
                  nn.BatchNorm2d(num_filter) if self.use_batch_norm else nn.Dropout(self.dropout),
              )
          )
      self.generator = nn.Linear(sum(self.num_filter_list), self.output_dim)

    def forward(self, data_dict, stage='train'):
        inputs = data_dict['input_ids']
        inputs = self.emb(inputs)
        min_length = max(self.window_size_list)
        if min_length > inputs.size(1):
            pad = inputs.new(inputs.size(0), min_length - inputs.size(1), self.embedding_dim).zero_()
            inputs = torch.cat([inputs, pad], dim=1)
        inputs = inputs.unsqueeze(1)

        cnn_outs = []
        for block in self.feature_extractors:
            cnn_out = block(inputs)
            cnn_out = nn.functional.max_pool1d(
                input=cnn_out.squeeze(-1),
                kernel_size=cnn_out.size(-2)
            ).squeeze(-1)
            cnn_outs += [cnn_out]
        cnn_outs = torch.cat(cnn_outs, dim=-1)
        
        output = self.generator(cnn_outs)
        output = F.softmax(output, dim=1)
        output_dict = {'output': output}
        return output_dict
