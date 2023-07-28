"""
사전훈련 모델이 사용할 데이터로더를 만드는 라이브러리.

주요 클래스:
    PM_Dataset: 입력 데이터를 받아 데이터로더에 전달할 데이터 집합을 만듭니다.
    PM_DataModule: 입력 데이터로 데이터세트와 데이터로더를 만듭니다.
"""


import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class PM_Dataset(Dataset):
    def __init__(self, df, config, stage='train'):
        self.df = df
        self.config = config
        self.max_len = config.max_len
        self.tokenizer = config.pm_tokenizer
        self.stage = stage
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        data = self.df.iloc[index]
        data_dict = self.tokenizer.encode_plus(
                                              data['document'],
                                              add_special_tokens=True,
                                              max_length=self.config.max_len,
                                              padding='max_length',
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                             )
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data_dict['input_ids'] = data_dict['input_ids'].squeeze(0).to(device)
        data_dict['attention_mask'] = data_dict['attention_mask'].squeeze(0).to(device)
        data_dict['token_type_ids'] = data_dict['token_type_ids'].squeeze(0).to(device)
        if self.stage != 'predict':
            data_dict['label'] = torch.tensor(data['label']).to(device)
        return data_dict


class PM_DataModule:
    def __init__(self, config, data):
        self.config = config
        # 학습과 검증 때는 2개의 데이터(학습, 검증 데이터)가 입력되고, 시험 때는 1개의 데이터만 입력.
        if isinstance(data, tuple):
            self.train_df, self.valid_df = data
        elif isinstance(data, pd.DataFrame):
            self.test_df = data
        self.tokenizer = config.pm_tokenizer
        self.vocab = None
    
    def build_dataset(self, df, stage='train'):
        dataset = PM_Dataset(df, self.config, stage=stage)
        return dataset

    def build_train_dataloader(self):
        dataset = self.build_dataset(self.train_df, stage='train')
        dataloader = DataLoader(dataset, batch_size=self.config.pm_batch_size, shuffle=True)
        return dataloader

    def build_valid_dataloader(self):
        dataset = self.build_dataset(self.valid_df, stage='valid')
        dataloader = DataLoader(dataset, batch_size=self.config.pm_batch_size, shuffle=False)
        return dataloader

    def build_test_dataloader(self):
        dataset = self.build_dataset(self.test_df, stage='test')
        dataloader = DataLoader(dataset, batch_size=self.config.pm_batch_size, shuffle=False)
        return dataloader
    
    def build_predict_dataloader(self):
        dataset = self.build_dataset(self.test_df, stage='predict')
        dataloader = DataLoader(dataset, batch_size=self.config.pm_batch_size, shuffle=False)
        return dataloader
