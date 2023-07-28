"""
딥러닝 모델이 사용할 데이터로더를 만드는 라이브러리.

주요 클래스:
    DL_Dataset: 토큰 리스트 집합과 라벨 집합을 입력 받아 각 토큰 리스트와 라벨을 텐서로 변환합니다.
    DL_DataModule: 입력 데이터로 토큰화, 단어집합 생성, 정수 인코딩, 패딩(선택사항) 작업을 진행하여 데이터세트와 데이터로더를 만듭니다.

주요 함수:
    collate_fn: RNN 계열에 입력할 데이터를 패킹 방식으로 구성할 때 배치 단위로 길이를 통일시켜 데이터로더에서 오류가 발생하지 않게 해줍니다.
"""


import pandas as pd
import torch
from torch import nn
from torchtext.vocab import vocab
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from collections import Counter, OrderedDict


def collate_fn(batch):
    """
    배치에 포함된 다양한 길이의 데이터를 배치에서 가장 긴 데이터의 길이로 통일시킵니다.

    Args:
        batch: 문장을 정수 인코딩한 데이터가 배치 크기만큼 담겨 있는 데이터 집합. tensor.
    Returns:
        동일한 길이로 맞춰진 인코딩 데이터, 길이가 통일되기 전의 각 인코딩 데이터 길이, 라벨. dictionary. 

    """
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
    """
    입력 데이터로 일련의 작업을 수행해 데이터로더를 만듭니다.

    Args:
        config: 미리 정의한 설정값. class.
        data: 입력 데이터. 2개의 dataframe을 담은 tuple 또는 dataframe.
    Returns:
        DataLoader.
    """
    def __init__(self, config, data=None):
        self.config = config
        # 학습과 검증 때는 2개의 데이터(학습, 검증 데이터)가 입력되고, 시험 때는 1개의 데이터만 입력.
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
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in self.config.stopwords]
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

        text_pipeline = lambda x: [self.vocab[token] for token in tokenizer(x)]
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
