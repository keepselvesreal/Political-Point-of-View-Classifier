import os
import sys
import glob
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
    fusion = True # 변수. fusion 여부에 따라 변경
    
    model = 'rnn' # 변수
    batch_size = 64 # 하이퍼파라미터 튜닝 대상
    epochs = 2 # 변수? # 하이퍼파라미터 튜닝 대상?
    load_data = False # 데이터로더 저장 유무
    tokenizer = Mecab()
    tokenizer_type = 'morphs'
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    min_freq = 3
 
    vocab = None
    vocab_size = None
    seq_type = 'packing' # 하이퍼파라미터 튜닝 대상
    max_len = 300 # 변수
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # 어디에 사용되는지 살펴보기

    dropout = 0.2 # 하이퍼파라미터 튜닝 대상
    output_dim = 4 
    embedding_dim = 256 # 하이퍼파라미터 튜닝 대상
    hidden_dim = 128 # 하이퍼파라미터 튜닝 대상
    rnn_type = 'GRU' # 하이퍼파라미터 튜닝 대상
    bidirectional = True # 하이퍼파라미터 튜닝 대상
    num_layers = 5 # 하이퍼파라미터 튜닝 대상
    activation = 'ReLU' # 하이퍼파라미터 튜닝 대상
    window_size = 5 # 하이퍼파라미터 튜닝 대상
    window_size_list = [2 ,4, 6] # 하이퍼파라미터 튜닝 대상
    num_filter = 100 # 하이퍼파라미터 튜닝 대상
    num_filter_list = [150, 150, 150] # 하이퍼파라미터 튜닝 대상
    use_batch_norm = False # 하이퍼파라미터 튜닝 대상

    loss = 'CrossEntropyLoss'
    lr = 1e-3
    lr_scheduler = 'gls' # # 하이퍼파라미터 튜닝 대상. gls, cos, exp 중 선택
    optim = 'Adam' # # 하이퍼파라미터 튜닝 대상. Adam, RMSprop, AdamP 중 선택
    warm_steps = 150 # # 하이퍼파라미터 튜닝 대상. 보통 전체 학습 스탭의 5~20%
    T_0 = 20 # # 하이퍼파라미터 튜닝 대상. 일반적으로 10~100 사이 값
    T_mult = 1 # # 하이퍼파라미터 튜닝 대상. 기본값=1
    eta_min = 1e-4 #  # 하이퍼파라미터 튜닝 대상. 기본값=0
    gamma = 0.5 # # 하이퍼파라미터 튜닝 대상. 기본값=0(학습률이 변하지 않음) 1보다 작으면 learning rate는 epoch에 따라 증가, 크면 동일한 방식으로 감소
    scaler = GradScaler() 
    max_grad_norm = False # 하이퍼파라미터 튜닝 대상
    accumulation_steps = False # 하이퍼파라미터 튜닝 대상
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

def transfer(model_path, new_data, model, id, ensemble=False):
    data = torch.load(model_path)

    vocab = data['vocab']
    CFG.vocab = vocab # test에는 vocab 저장하지 않아도 되므로 주석 처리 가능
    datamodule = DL_DataModule(CFG, new_data)
    datamodule.vocab = vocab
    train_dataloader = datamodule.build_train_dataloader()
    valid_dataloader = datamodule.build_valid_dataloader()
    print() 

    CFG.vocab_size = len(vocab)
    model = model(CFG)
    state_dict = data['state_dict']
    model.load_state_dict(state_dict)

    trainer = Trainer(CFG, model, (train_dataloader, valid_dataloader), is_pm=False)
    outputs_dict = trainer.fit()
    outputs_dict['content'] = new_data[1].document.values

    print()
    summarize_result(CFG, id, outputs_dict, 'test')
    print()
    show_confusion_matrix(CFG, id, outputs_dict, 'test')
    
    wandb.finish()

    if ensemble:
        return model.__class__.__name__, outputs_dict
       
def main(args):
    CFG.model = args.model
    CFG.epochs = args.nepochs
    CFG.fold = args.fold
    CFG.fusion = args.fusion
    file_name = args.file_name
    CFG.csv_path = f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}' if CFG.fusion else f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}'
    CFG.model_path = '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/base_models/'

    id = initialize_wandb((RNN_CONFIG, CNN_CONFIG), args, 'train')

    train_df = pd.read_csv(CFG.csv_path)
    if CFG.stratified:
        folds = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True)
        train_idx, valid_idx = list(folds.split(train_df.document, train_df.label))[CFG.fold]
    else:
        folds = KFold(n_splits=5, shuffle=True)
        train_idx, valid_idx = list(folds.split(train_df.values))[CFG.fold]
    valid_df = train_df.iloc[valid_idx] # train_df부터 하면 밑에 train_df도 변해서 인덱스 오류 발생
    train_df = train_df.iloc[train_idx]
    print('train_df shape', train_df.shape)
    print('valid_df shape', valid_df.shape)
    print()

    if CFG.model == 'rnn':
        rnn_model_path_list = glob.glob(os.path.join(CFG.model_path, '*Rnn*'))
        print('RNN model path list -> ', rnn_model_path_list)
        selected_model_path = rnn_model_path_list[CFG.fold]
        print('selected model path -> ', selected_model_path)
        print()
        transfer(selected_model_path, (train_df, valid_df), Rnn, id)
    elif CFG.model == 'cnn':
        cnn_model_path_list = glob.glob(os.path.join(CFG.model_path, '*Cnn*'))
        print('CNN model path list-> ', cnn_model_path_list)
        selected_model_path = cnn_model_path_list[CFG.fold]
        print('selected model path -> ', selected_model_path)
        print()
        transfer(selected_model_path, (train_df, valid_df), Cnn, id)

if __name__ == '__main__':
    main(args)
