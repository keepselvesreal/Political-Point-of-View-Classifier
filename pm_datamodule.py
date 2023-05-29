import torch
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
    def __init__(self, config, df):
        self.config = config
        if len(df) == 2:
            self.train_df, self.valid_df = df
        else:
            self.test_df = df
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