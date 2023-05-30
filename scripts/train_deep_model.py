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
    stratified = True
    n_splits = 5
    fold = 0 # 변수
    fusion = True # False: <온라인 커뮤니티 데이터>로 학습하는 경우/ True : <온라인 커뮤니티 데이터 + 네이버 댓글 데이터>로 학습하는 경우
    
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
    lr_scheduler = 'gls'  # 'gls': get_linear_schedule_with_warmup, 'cos': CosineAnnealingWarmRestarts, 'exp': ExponentialLR, 'red': ReduceLROnPlateau, 'step' StepLR
    optim = 'Adam' # 'Adam', 'RMSprop', 'AdamP'
    warm_steps = 150
    T_0 = 20 
    T_mult = 1
    eta_min = 1e-4 
    gamma = 0.5
    scaler = GradScaler() 
    max_grad_norm = False 
    accumulation_steps = False
    patience = 3
    
    # extra.csv(transfer_train data)=community+naver data, train_data=community_data
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
    CFG.csv_path = f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}' if CFG.fusion else f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}'
    CFG.model_path = '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/fusion_models/' if CFG.fusion else '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/base_models/'

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
    outputs_dict['content'] = valid_df.document.values

    print()
    summarize_result(CFG, id, outputs_dict, 'train')
    print()
    show_confusion_matrix(CFG, id, outputs_dict, 'train')
    
    wandb.finish(quiet=True)

if __name__ == '__main__':
    main(args)
