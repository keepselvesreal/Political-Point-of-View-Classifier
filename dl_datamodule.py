import pandas as pd
import torch
from torch import nn
from torchtext.vocab import vocab
from torch.utils.data import DataLoader, Dataset

from tqdm.notebook import tqdm
from collections import Counter, OrderedDict

class CFG:
    vocab = None
    vocab_size = None

def collate_fn(batch):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_ids = nn.utils.rnn.pad_sequence([data_dict['input_ids'] for data_dict in batch], batch_first=True).to(device)
    length = torch.tensor([len(data_dict['input_ids']) for data_dict in batch]).to(device)
    
    if 'label' in batch[0]:
        label = torch.tensor([data_dict['label'] for data_dict in batch]).to(device)
        return {'input_ids': input_ids, 'label': label, 'length': length}
    else:
        return {'input_ids': input_ids, 'length': length}

class DL_Dataset(Dataset):
  def __init__(self, token_list, label_list=None, stage='train'):
    self.token_list = token_list
    self.label_list = label_list
    self.stage=stage

  def __getitem__(self, idx):
    data_dict = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if self.stage == 'predict':
        data_dict['input_ids'] = torch.tensor(self.token_list[idx]).to(device)
    else:
        data_dict['input_ids'], data_dict['label'] = torch.tensor(self.token_list[idx]).to(device), torch.tensor(self.label_list[idx]).to(device)
    return data_dict

  def __len__(self):
    return len(self.token_list)
  

class DL_DataModule:
    def __init__(self, config, data=None):
        self.config = config
        if isinstance(data, tuple):
            self.train_df, self.valid_df = data
        elif isinstance(data, pd.DataFrame):
            self.test_df = data
        self.tokenizer = config.tokenizer
        self.vocab = None
        self.vocab_size = None

    def build_vocab(self, df):
        token_counts = Counter()
        for row in tqdm(df.itertuples(), total=len(df), desc='tokenizing', leave=True):
            tokenized_sentence = self.tokenizer.morphs(row.document)
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in self.config.stopwords] # 불용어 제거
            token_counts.update(stopwords_removed_sentence)

        filtered_token_counts = dict(filter(lambda x: x[1] > self.config.min_freq, token_counts.items()))
        sorted_token_counts = sorted(filtered_token_counts.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_token_counts)

        vocabulary = vocab(ordered_dict)
        vocabulary.insert_token('<pad>', 0)
        vocabulary.insert_token('<unk>', 1)
        vocabulary.set_default_index(1)
        self.vocab = vocabulary
        self.vocab_size = len(vocabulary)

    def encode(self, df, stage='train'):
        if self.config.tokenizer_type == 'morphs':
            tokenizer = self.tokenizer.morphs
        elif self.confi.tokenizer_type == 'nouns':
            tokenizer = self.tokenizer.nouns

        text_pipeline = lambda x: [self.vocab[token] for token in tokenizer(x)] # 221126 version에서 stem=True 제거
        token_list = []
        drop_idx_list = []
        for row in tqdm(df.itertuples(), total=len(df), desc='encoding', leave=True):
            tokens = text_pipeline(row.document)
            if len(tokens) > 1:
                token_list.append(tokens)
            else:
                drop_idx_list.append(row[0])
        df.drop(index=drop_idx_list, inplace=True)
        if stage == 'predict':
            return token_list
        else:
            label_list = df['label'].to_list()
            return token_list, label_list            

    def pad(self, token_list):
        padding_list = []
        for token in tqdm(token_list, total=len(token_list), desc='padding'):
            token = token[:self.config.max_len]
            length = len(token)
            if length < self.config.max_len:
                token += [self.vocab['<pad>']] * (self.config.max_len - length)
            padding_list.append(token)
        return padding_list

    def save(self, object, name):
        path = self.config.data_path     
        torch.save(object, path + name)
        
    def load(self, name):
        path = self.config.data_path
        object = torch.load(path + name)
        return object

    def build_dataset(self, df, stage='train'):
        if stage == 'train':
            if not self.vocab:
                self.build_vocab(df)

        if stage == 'predict':
            token_list = self.encode(df, stage=stage)
            if self.config.seq_type == 'padding':
                token_list = self.pad(token_list)
            dataset = DL_Dataset(token_list, stage=stage)
        else:
            token_list, label_list = self.encode(df, self.vocab)
            if self.config.seq_type == 'padding':
                token_list = self.pad(token_list)
            dataset = DL_Dataset(token_list, label_list, stage=stage)
        return dataset

    def build_dataloader(self, dataset, batch_size, shuffle, stage='train'):
        if self.config.seq_type == 'packing':
            dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        setattr(self, f'{stage}_dataloader', dataloader)
        return dataloader

    def build_train_dataloader(self):
        if self.config.load_data:
            return self.load_dataloader(stage='train')
        dataset = self.build_dataset(self.train_df, stage='train')
        dataloader = self.build_dataloader(dataset, batch_size=self.config.batch_size, shuffle=True, stage='train')
        return dataloader

    def build_valid_dataloader(self):
        if self.config.load_data:
            return self.load_dataloader('valid')
        dataset = self.build_dataset(self.valid_df, stage='valid')
        dataloader = self.build_dataloader(dataset, batch_size=self.config.batch_size, shuffle=False, stage='valid')
        return dataloader

    def build_test_dataloader(self):
        if self.config.load_data:
            return self.load_dataloader('test')
        dataset = self.build_dataset(self.test_df, stage='test')
        dataloader = self.build_dataloader(dataset, batch_size=self.config.batch_size, shuffle=False, stage='test')
        return dataloader
    
    def build_predict_dataloader(self):
        if self.config.load_data:
            return self.load_dataloader('predict')
        dataset = self.build_dataset(self.test_df, stage='predict')
        dataloader = self.build_dataloader(dataset, batch_size=self.config.batch_size, shuffle=False, stage='predict')
        return dataloader