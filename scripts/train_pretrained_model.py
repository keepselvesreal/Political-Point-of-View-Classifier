import os
import sys
import random
import wandb
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import AutoTokenizer
from torch.cuda.amp import GradScaler

sys.path.append('cd /content/drive/MyDrive/프로젝트/politic_value_relationship/test3/files')
from pm_datamodule import PM_DataModule
from trainer import Trainer
from pm_model import Kcbert
from visualizer import Visualizer
from utils import initialize_wandb, summarize_result, show_confusion_matrix

class CFG:
    seed = 7
    stratified = True
    n_splits = 5
    fold = 0
    fusion = False # False: <온라인 커뮤니티 데이터>로 학습하는 경우/ True : <온라인 커뮤니티 데이터 + 네이버 댓글 데이터>로 학습하는 경우
    
    model = 'Kcbert'
    pm_batch_size = 16
    epochs = 2
    load_data = False
    vocab = None
    max_len = 300 
    output_dim = 4
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    pm_model_name = 'beomi/kcbert-base'
    pm_tokenizer = AutoTokenizer.from_pretrained(pm_model_name)

    loss = 'CrossEntropyLoss'
    lr = 1e-3
    pm_lr = 5e-6
    lr_scheduler = 'gls'  # 'gls': get_linear_schedule_with_warmup, 'cos': CosineAnnealingWarmRestarts, 'exp': ExponentialLR, 'red': ReduceLROnPlateau, 'step' StepLR
    optim = 'Adam' # 'Adam', 'RMSprop', 'AdamP'
    warm_steps = 100 
    T_0 = 20 
    T_mult = 1 
    eta_min = 1e-4 
    gamma = 0.5
    scaler = GradScaler()
    max_grad_norm = False 
    accumulation_steps = False
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

WANDB_CONFIG = {
    'model': CFG.model,
    'fold': CFG.fold,
    'arch': 'PM',
    'data': None,
    
    'epochs': CFG.epochs,
    'batch_size': CFG.pm_batch_size,

    'optim': CFG.optim,
    'lr': CFG.pm_lr,
    'lr_scheduler': CFG.lr_scheduler,
    }

def main(args):
    CFG.model = args.model
    CFG.epochs = args.nepochs
    CFG.fold = args.fold
    CFG.fusion = args.fusion
    file_name = args.file_name
    CFG.csv_path = f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}' if CFG.fusion else f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}'
    CFG.model_path = '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/fusion_models/' if CFG.fusion else '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/base_models/'

    id = initialize_wandb(WANDB_CONFIG, args, 'train')

    train_df = pd.read_csv(CFG.csv_path)
    if CFG.stratified:
        folds = StratifiedKFold(n_splits=5, random_state=CFG.seed, shuffle=True)
        train_idx, valid_idx = list(folds.split(train_df.document,
                                                train_df.label))[CFG.fold]
    else:
        folds = KFold(n_splits=5, random_state=CFG.seed, shuffle=True)
        train_idx, valid_idx = list(folds.split(train_df.values))[CFG.fold]
    valid_df = train_df.iloc[valid_idx]
    train_df = train_df.iloc[train_idx]
    print('train_df shape', train_df.shape)
    print('valid_df shape', valid_df.shape)

    datamodule = PM_DataModule(CFG, (train_df, valid_df))
    train_dataloader = datamodule.build_train_dataloader()
    valid_dataloader = datamodule.build_valid_dataloader()

    if CFG.model == 'Kcbert':
        model = Kcbert(CFG)
    trainer = Trainer(CFG, model, (train_dataloader, valid_dataloader), is_pm=True)
    outputs_dict = trainer.fit()
    outputs_dict['content'] = valid_df.document.values
    
    summarize_result(CFG, id, outputs_dict, 'train')
    show_confusion_matrix(CFG, id, outputs_dict, 'train')
    attention_dict = trainer.attention_dict
    visualizer = Visualizer(CFG, None, id, attention_dict)
    visualizer.show_attention()
    output_path = CFG.output_path + 'train'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path + f'/train_visualizer_{id}', 'wb') as f:
        pickle.dump(visualizer, f)

    wandb.finish(quiet=True)

if __name__  == '__main__':
    main(args)




