"""
사전훈련모델을 구성하는 라이브러리.
"""


from torch import nn
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
        # 각 토큰에 부여된 어텐션 스코어를 시각화하는 데 사용하기 위해 모델 검증과 시험 때에는 마지막 블록의 어텐션 점수도 수집.  
        else:
            output_dict = {'output': output, 'attn_prob': attn_probs[-1]}
            return output_dict
