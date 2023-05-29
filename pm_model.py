import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

class Kcbert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.pm_model_name)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 4)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, data_dict, stage='train'):
        input_ids, attention_mask, token_type_ids = data_dict['input_ids'], data_dict['attention_mask'], data_dict['token_type_ids']
        _, last_hidden_states, attn_probs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=True, return_dict=False)
        last_hidden_states = self.drop(last_hidden_states)
        last_hidden_states = self.fc(last_hidden_states)
        output = self.softmax(last_hidden_states)
        if stage == 'train':
            output_dict = {'output': output} 
            return output_dict
        else:
            output_dict = {'output': output, 'attn_prob': attn_probs[-1]}
            return output_dict