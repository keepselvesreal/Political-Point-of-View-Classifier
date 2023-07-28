"""
딥러닝 모델을 훈련시키는 모듈

데이터를 불러와 훈련, 검증 데이터를 만든 후 dl_datamodule, dl_model, trainer 모듈을 이용해 모델을 훈련하고 검증합니다.
훈련과 검증 과정의 모델 성능 지표는 wandb 라이브러리를 이용해 기록합니다.
모델의 반환값 모두를 파일에 저장하고, 주요 반환값을 화면에 출력하며, 혼동 행렬도 시각화합니다. 
"""


import os
import sys
import random
import wandb
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from sklearn.model_selection import StratifiedKFold, KFold
from konlpy.tag import Mecab

sys.path.append('cd /content/drive/MyDrive/프로젝트/politic_value_relationship/test3/files')
from dl_datamodule import DL_DataModule
from trainer import Trainer
from dl_model import Rnn, Cnn
from utils import initialize_wandb, summarize_result, show_confusion_matrix

class CFG:
    seed = 7
    stratified = True # 사이킷런의 StratifiedKFold를 사용할지 선택하는 설정값. False로 선택 시 KFold을 사용.
    n_splits = 5
    fold = 0 # 사이킷런의 StratifiedKFold로 만들어진 folds 중 하나를 선택하는 인덱스.
    fusion = True # False: <온라인 커뮤니티 데이터>로 학습하는 경우. True : <온라인 커뮤니티 데이터 + 네이버 댓글 데이터>로 학습하는 경우.
    
    model = None
    batch_size = 64 
    epochs = 7 
    load_data = False 
    tokenizer = Mecab()
    tokenizer_type = 'morphs'
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    min_freq = 3

    vocab = None
    vocab_size = None
    seq_type = 'packing'
    max_len = 300
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    dropout = 0.2
    output_dim = 4 
    embedding_dim = 256 
    hidden_dim = 128 
    rnn_type = 'GRU' 
    bidirectional = True 
    num_layers = 5 #
    activation = 'ReLU' 
    window_size = 5 
    window_size_list = [2 ,4, 6] 
    num_filter = 100 
    num_filter_list = [150, 150, 150]
    use_batch_norm = False

    loss = 'CrossEntropyLoss'
    lr = 1e-3
    lr_scheduler = 'gls'  # 다음 옵션 중 선택. 'gls'=get_linear_schedule_with_warmup, 'cos'=CosineAnnealingWarmRestarts, 'exp'=ExponentialLR,'red'=ReduceLROnPlateau, 'step'=StepLR
    optim = 'Adam' # 다음 옵션 중 선택. 'Adam', 'RMSprop', 'AdamP'
    warm_steps = 150 #
    T_0 = 20 #
    T_mult = 1 #
    eta_min = 1e-4 # 
    gamma = 0.5 #
    scaler = GradScaler() # 
    max_grad_norm = False # gradient clipping 사용 여부를 결정하는 설정값.
    accumulation_steps = False # gradient accumulation 사용 여부를 결정하는 설정값.
    patience = 3
    
    csv_path =  None
    model_path = None
    output_path = '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/outputs/'

os.environ['PYTHONHASHSEED'] = str(CFG.seed)
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)    
torch.cuda.manual_seed(CFG.seed)
torch.backends.cudnn.deterministic = True

wandb.login(key='ed8eccb1b48e18c315c20244f4b7156c28be8ec0')

RNN_CONFIG = {
    'model': CFG.model,
    'fold': CFG.fold,
    'arch': 'DL',
    'data': None,
    
    'epochs': CFG.epochs,
    'batch_size': CFG.batch_size,

    'optim': CFG.optim,
    'lr': CFG.lr,
    'lr_scheduler': CFG.lr_scheduler,

    'rnn_type': CFG.rnn_type,
    'embedding_dim': CFG.embedding_dim,
    'hidden_dim': CFG.hidden_dim,    
    'num_layers': CFG.num_layers,
    'bidirectional': CFG.bidirectional,
    'dropout': CFG.dropout
}

CNN_CONFIG = {
    'model': CFG.model,
    'fold': CFG.fold,
    'arch': 'DL',
    'data': None,

    'epochs': CFG.epochs,
    'batch_size': CFG.batch_size,
    'max_len': CFG.max_len,

    'optim': CFG.optim,
    'lr': CFG.lr,
    'lr_scheduler': CFG.lr_scheduler,

    'window_size_list': CFG.window_size_list,
    'num_filter_list': CFG.num_filter_list,
    'dropout': CFG.dropout,
    'use_batch_norm': CFG.use_batch_norm
}

def main(args):
    CFG.model = args.model
    CFG.epochs = args.nepochs
    CFG.fold = args.fold
    CFG.fusion = args.fusion
    file_name = args.file_name
    # CFG.fusion=True인 경우: <온라인 커뮤니티 데이터 + 네이버 댓글 데이터>로 구성된 데이터의 경로를 선택. CFG.fusion=False 경우: <온라인 커뮤니티 데이터>의 경로를 선택.
    CFG.csv_path = f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}' if CFG.fusion else f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}'
    # CFG.fusion=True인 경우: <온라인 커뮤니티 데이터 + 네이버 댓글 데이터>로 훈련한 모델이 저장되는 경로를 선택. CFG.fusion=False 경우: <온라인 커뮤니티 데이터>로 훈련한 모델이 저장되는 경로를 선택.
    CFG.model_path = '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/fusion_models/' if CFG.fusion else '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/base_models/'

    # utils 모듈의 initialize_wandb 함수를 이용해 wandb 설정값을 입력하고 기록을 시작. wandb 기록 확인에 사용할 식별자(str)를 반환.
    id = initialize_wandb((RNN_CONFIG, CNN_CONFIG), args, 'train')

    train_df = pd.read_csv(CFG.csv_path)
    if CFG.stratified:
        folds = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True)
        train_idx, valid_idx = list(folds.split(train_df.document, train_df.label))[CFG.fold]
    else:
        folds = KFold(n_splits=5, shuffle=True)
        train_idx, valid_idx = list(folds.split(train_df.values))[CFG.fold]
    valid_df = train_df.iloc[valid_idx]
    train_df = train_df.iloc[train_idx]
    print('train_df shape', train_df.shape)
    print('valid_df shape', valid_df.shape)
    print()

    datamodule = DL_DataModule(CFG, (train_df, valid_df))
    train_dataloader = datamodule.build_train_dataloader()
    valid_dataloader = datamodule.build_valid_dataloader()
    print()
    CFG.vocab = datamodule.vocab
    CFG.vocab_size = datamodule.vocab_size

    if CFG.model == 'rnn':
        model = Rnn(CFG)
    elif CFG.model == 'cnn':
        model = Cnn(CFG)

    trainer = Trainer(CFG, model, (train_dataloader, valid_dataloader))
    outputs_dict = trainer.fit()
    # summarize_result 함수에서 사용하기 위해 입력 텍스트를 수집.
    outputs_dict['content'] = valid_df.document.values

    print()
    # utils 모듈의 summarize_result 함수를 이용해 모델의 반환값을 파일에 모두 저장한 후 loss를 기준으로 상위 10개, 하위 10개의 모델 예측과 라벨을 화면에 출력.
    summarize_result(CFG, id, outputs_dict, 'train')
    print()
    # utils 모듈의 show_confusion_matrix 함수를 이용해 혼동 행렬을 화면에 출력.
    show_confusion_matrix(CFG, id, outputs_dict, 'train')
    
    wandb.finish(quiet=True)

if __name__ == '__main__':
    main(args)
