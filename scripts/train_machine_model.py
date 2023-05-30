import os 
import sys
import random
import wandb
import pandas as pd
import numpy as np
from konlpy.tag import Mecab
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

sys.path.append('cd /content/drive/MyDrive/프로젝트/politic_value_relationship/test3/files')
from ml_trainer import ML_Trainer
from utils import initialize_wandb, summarize_result, show_confusion_matrix

class CFG:
    seed = 7
    stratified = True
    n_splits = 5
    fusion = False # False: <온라인 커뮤니티 데이터>로 학습하는 경우/ True : <온라인 커뮤니티 데이터 + 네이버 댓글 데이터>로 학습하는 경우
    fold = 0 # 변수

    model = None
    params = None 
    cv = 5 
    n_iter = 20
    use_gbm = True if model == 'LGBM' else False
    
    tokenizer = Mecab()
    tokenizer_type = 'morphs'
    max_len = 300
    
    csv_path =  None
    model_path = None
    output_path = '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/outputs/'

os.environ['PYTHONHASHSEED'] = str(CFG.seed)
random.seed(CFG.seed)
np.random.seed(CFG.seed)

wandb.login(key='ed8eccb1b48e18c315c20244f4b7156c28be8ec0')

WANDB_CONFIG = {
        'model': CFG.model,
        'fold': CFG.fold,
        'arch': 'ML',
        'data': None
        }

def main(args):
  CFG.model = args.model
  CFG.fold = args.fold
  CFG.fusion = args.fusion
  file_name = args.file_name
  CFG.csv_path = f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}' if CFG.fusion else f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}'
  CFG.model_path = '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/fusion_models/' if CFG.fusion else '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/base_models/'

  id = initialize_wandb(WANDB_CONFIG, args, 'train')

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

  if CFG.model == 'LR':
    model = LogisticRegression(multi_class='multinomial')
  elif CFG.model == 'NB':
    model = MultinomialNB(alpha=0.1)
  elif CFG.model == 'SVC':
    model = SVC(probability=True)
  elif CFG.model == "LGBM":
    model = LGBMClassifier(learning_rate=0.1, 
                          max_depth=8,
                          min_split_gain=4,
                          min_child_weight=5,
                          subsample=0.1,
                          colsample_bytree=0.5,
                          objective='multiclass')
  
  ml_trainer = ML_Trainer(CFG, (train_df, valid_df), model, stage='train') 
  params = CFG.params
  outputs_dict = ml_trainer.fit(params=params)
  outputs_dict['content'] = valid_df.document.values

  print()
  summarize_result(CFG, id, outputs_dict, 'train')
  print()
  show_confusion_matrix(CFG, id, outputs_dict, 'train')

  wandb.finish(quiet=True)

if __name__ == '__main__':
   main(args)
