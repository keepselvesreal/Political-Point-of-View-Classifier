"""
훈련된 서로 다른 종류의 모델들로 소프트 보팅을 수행하는 라이브러리.

시험 데이터를 불러온 후 각 모델 유형별로 훈련된 모델을 임의로 하나씩 선택하고 소프트 보팅을 수행합니다.
시험 과정의 모델 성능 지표는 wandb 라이브러리를 이용해 기록합니다.
선별한 데이터에 대한 소프트 보팅 예측과 라벨을 화면에 출력하고, 혼동 행렬도 시각화합니다.
소프트 보팅에 참여한 각 모델의 예측, 라벨, 정확도를 화면에 출력하고, 모든 예측(개별 모델의 예측 + 소프트 보팅의 예측)을 대상으로 계산한 각 예측 간 상관관계도 시각화합니다. 

주요 함수:
  ml_test: 임의로 선택한 전통적인 머신러닝 모델로 시험 데이터에 대한 예측을 수행합니다.
  dl_test: 임의로 선택한 커스텀 딥러닝 모델로 시험 데이터에 대한 예측을 수행합니다.
  pl_test: 임의로 선택한 사전훈련 모델로 시험 데이터에 대한 예측을 수행합니다.
  ensemble: 임의로 선택한 서로 다른 종류의 훈련 모델들로 소프트 보팅을 수행합니다.
"""


import os
import sys
import random
import glob
import wandb
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from konlpy.tag import Mecab
from transformers import AutoTokenizer
from IPython.display import HTML, display

sys.path.append('cd /content/drive/MyDrive/프로젝트/politic_value_relationship/test3/files')
from dl_datamodule import DL_DataModule
from pm_datamodule import PM_DataModule
from ml_trainer import ML_Trainer
from trainer import Trainer
from dl_model import Rnn, Cnn
from pm_model import Kcbert
from utils import initialize_wandb, get_metrics, log_metrics, show_confusion_matrix
from visualizer import int2str

class CFG:
  seed = 7
  fusion = False # False: <온라인 커뮤니티 데이터>로 학습한 모델을 사용/ True : <온라인 커뮤니티 데이터 + 네이버 댓글 데이터>로 학습한 모델을 사용

  batch_size = 64
  pm_batch_size = 16
  
  tokenizer = Mecab()
  tokenizer_type = 'morphs'
  load_data = False
  max_len = 300
  seq_type = 'packing'
  vocab = None
  vocab_size = None
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

  pm_model_name = 'beomi/kcbert-base'
  pm_tokenizer = AutoTokenizer.from_pretrained(pm_model_name)

  output_dim = 4 
  rnn_type = 'GRU'
  embedding_dim = 256
  hidden_dim = 128
  num_layers = 5
  bidirectional = True
  activation = 'ReLU'
  window_size = 5
  window_size_list = [2 ,4, 6]
  num_filter = 100
  num_filter_list = [150, 150, 150] 
  use_batch_norm = False
  dropout = 0.2

  csv_path = None
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
    'model': 'multiple model',
    'fold': 'random',
    'arch': 'ML, DL, PM',
    'data': None
    }

def ml_test(model_path, test_df):
  """
  임의로 선택한 전통적인 머신러닝 모델로 시험 데이터에 대한 예측을 수행합니다.
  """
  ml_trainer = ML_Trainer(CFG, test_df, stage='test')
  ml_trainer.load(model_path)
  model_name = ml_trainer.model.__class__.__name__
  outputs_dict = ml_trainer.test()
  return model_name, outputs_dict

def dl_test(model_path, test_df, model):
  """
  임의로 선택한 커스텀 딥러닝 모델로 시험 데이터에 대한 예측을 수행합니다.
  """
  data = torch.load(model_path)

  datamodule = DL_DataModule(CFG, test_df)
  # 저장된 데이터에서 모델 훈련 때 생성한 사전을 가져옴.
  vocab = data['vocab']
  # 데이터모듈 객체에 위에서 가져온 사전을 전달.
  datamodule.vocab = vocab
  test_dataloader = datamodule.build_test_dataloader()
  print()   

  # 딥러닝 모델의 임베딩 층을 만드는 데 필요한 토큰 개수 정보 저장.
  CFG.vocab_size = len(vocab)
  model = model(CFG)
  state_dict = data['state_dict']
  model.load_state_dict(state_dict)
  model_name = model.__class__.__name__

  trainer = Trainer(CFG, model, test_dataloader, is_pm=False)
  outputs_dict = trainer.test()
  return model_name, outputs_dict

def pm_test(model_path, test_df, model):
  """
  pl_test: 임의로 선택한 사전훈련 딥러닝 모델로 시험 데이터에 대한 예측을 수행합니다.
  """
  data = torch.load(model_path) 

  model = model(CFG)
  state_dict = data['state_dict']
  model.load_state_dict(state_dict)
  model_name = model.__class__.__name__

  datamodule = PM_DataModule(CFG, test_df)
  test_dataloader = datamodule.build_test_dataloader()
  
  trainer = Trainer(CFG, model, test_dataloader, is_pm=True)
  outputs_dict = trainer.test()
  print()
  return model_name, outputs_dict

def ensemble(test_df, outputs_list, stage='test'):
  """
  임의로 선택한 서로 다른 종류의 훈련 모델들로 소프트 보팅을 수행합니다.
  """
  outputs_dict = {}
  probas = 0 
  for _, outputs in outputs_list:
      probas += np.array(outputs['output'])
  preds = np.argmax(probas, 1)
  outputs_dict['output'] = probas
  outputs_dict['pred'] = preds
  outputs_dict['content'] = test_df.document
  if stage == 'test':
    outputs_dict['target'] = test_df.label.values
  return outputs_dict

def inspect_ensemble(df, outputs_dict, stage='test'):
  """
  소프트 보팅 결과를 입력받아 loss를 기준으로 상위 10개, 하위 10개 데이터를 선별하고 최종 예측결과를 화면에 출력합니다.
  """
  input_data = df.copy()
  probas, preds = outputs_dict['output'], outputs_dict['pred']
  reshaped_preds = preds.reshape(-1, 1)
  # 각 클래스에 대한 예측 확률과 최종 예측 클래스를 결합.
  inspect_result = np.concatenate([probas, reshaped_preds], axis=1)
  inspect_result = pd.DataFrame(inspect_result, columns=['국힘갤류', '뽐뿌류', '펨코류', '루리웹류', '최종 예측'])
  # 함께 출력하기 위해 입력 텍스트도 결합.
  inspect_result = pd.concat([input_data, inspect_result], axis=1)
  if stage == 'test':
    inspect_result['loss'] = inspect_result.apply(lambda row: np.sqrt(np.square(row['label']-row['최종 예측'])), axis=1)
    inspect_result['label'] = inspect_result['label'].astype('int').apply(int2str)
    inspect_result['최종 예측'] = inspect_result['최종 예측'].apply(int2str)
    good_preds = inspect_result[inspect_result['최종 예측'] == inspect_result['label']].sort_values('loss').head(10).drop(columns='loss')
    bad_preds = inspect_result[inspect_result['최종 예측'] != inspect_result['label']].sort_values('loss').tail(10).drop(columns='loss')
    print('='*50 + '정답 예측 & loss 낮음 TOP10' + '='*50)
    display(HTML(good_preds.to_html()))
    print('='*50 + '오답 예측 & loss 높음 TOP10' + '='*50)
    display(HTML(bad_preds.to_html()))
  else:
    confi_preds = inspect_result.sort_values('최종 예측').tail(10)
    unconfi_preds = inspect_result.sort_values('최종 예측').head(10)
    inspect_result['최종 예측'] = inspect_result['최종 예측'].apply(int2str)
    print('<예측 확신 상위10>')
    display(HTML(confi_preds.to_html()))
    print('<예측 확신 하위10>')
    display(HTML(unconfi_preds.to_html()))
  return  inspect_result

def compare_preds(outputs_list, outputs_dict, stage='test'):
  """
  모든 예측(개별 모델의 예측 + 소프트 보팅의 예측)의 정확도를 계산하여 화면에 출력합니다.
  """
  model_preds = {model_name: np.argmax(outputs['output'], 1) for model_name, outputs in outputs_list}
  indv_model_pred = pd.DataFrame(model_preds)
  indv_model_pred['ensemble'] = outputs_dict['pred']
  if stage == 'test':
    indv_model_pred['target'] = outputs_dict['target']
    model_acc = round(indv_model_pred.eq(indv_model_pred['target'], axis=0).mean() * 100, 2)
    model_acc = model_acc.iloc[:-1]
    indv_model_acc = pd.DataFrame({'accuracy': model_acc})
    display(indv_model_acc)
    return indv_model_acc
  else:
    indv_model_pred = indv_model_pred.applymap(int2str)
    indv_model_pred = pd.concat([outputs_dict['content'], indv_model_pred], axis=1)
    display(indv_model_pred)
    return indv_model_pred

def show_corr(id, outputs_list, outputs_dict, stage='test'):
  """
  모든 예측(개별 모델의 예측 + 소프트 보팅의 예측)을 대상으로 상관관계를 계산하여 화면에 출력합니다.
  """
  model_preds = {model_name: np.argmax(output_dict['output'], 1) for model_name, output_dict in outputs_list}
  model_preds['ensemble'] = outputs_dict['pred']
  df = pd.DataFrame(model_preds)
  corr = df.corr()
  mask = np.zeros_like(corr, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True
  f, ax = plt.subplots(figsize=(11, 9))
  sns.heatmap(corr,
              annot=True,
              annot_kws={"size": 17},
              cmap='RdYlBu_r',
              mask=mask,
              linewidths=.5,
              cbar_kws={"shrink": .5},
              vmin = -1,vmax = 1,
              square=True,
              ax=ax)
  ax.tick_params(axis='both', labelsize=20, rotation=45)
  output_path = CFG.output_path + 'test'
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  plt.savefig(output_path + f'/{stage}_{id}_corr.png')
  plt.show()
  return (corr, mask)

def main(args):
    CFG.fusion = args.fusion
    file_name = args.file_name
    CFG.csv_path = f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}'
    CFG.model_path = '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/fusion_models/' if CFG.fusion else '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/base_models/'

    id = initialize_wandb(WANDB_CONFIG, args, 'test')

    test_df = pd.read_csv(CFG.csv_path)
    print('test_df shape', test_df.shape)

    rnn_model_list = glob.glob(os.path.join(CFG.model_path, '*Rnn*'))
    rnn_model_path = random.choice(rnn_model_list)
    print('RNN model list -> ', rnn_model_list)
    print('RNN model path -> ', rnn_model_path)
    
    cnn_model_list = glob.glob(os.path.join(CFG.model_path, '*Cnn*'))
    cnn_model_path = random.choice(cnn_model_list)
    print('CNN model list -> ', cnn_model_list)
    print('CNN model path -> ', cnn_model_path)
    
    pm_model_list = glob.glob(os.path.join(CFG.model_path, 'P*'))
    pm_model_path = random.choice(pm_model_list)
    print('PM model list -> ', pm_model_list)
    print('PM model path -> ', pm_model_path)

    lr_model_list = glob.glob(os.path.join(CFG.model_path, '*LR*'))
    lr_model_path = random.choice(lr_model_list)
    print('lr model list -> ', lr_model_list)
    print('lr model path -> ', lr_model_path)
    
    nb_model_list = glob.glob(os.path.join(CFG.model_path, '*NB*'))
    nb_model_path = random.choice(nb_model_list)
    print('nb model list -> ', nb_model_list)
    print('nb model path -> ', nb_model_path)

    # 모델들의 예측 결과를 수집. 
    # 딥러닝 모델을 위한 데이터 전처리 과정에서 입력 데이터 일부가 제거될 수 있어서 딥러닝 모델의 예측 결과를 전통적인 머신러닝 모델의 예측 결과보다 먼저 수집.
    outputs_list = []
    outputs_list.append(dl_test(rnn_model_path, test_df, Rnn)) 
    outputs_list.append(dl_test(cnn_model_path, test_df, Cnn))
    outputs_list.append(pm_test(pm_model_path, test_df, Kcbert))
    outputs_list.append(ml_test(lr_model_path, test_df))
    outputs_list.append(ml_test(nb_model_path, test_df))

    # 모델들의 예측 결과로 소프트 보팅 수행.
    outputs_dict = ensemble(test_df, outputs_list)
    metrics = get_metrics(outputs_dict['target'], outputs_dict['pred'])
    log_metrics(*metrics, stage='valid')
    
    # loss를 기준으로 상위 10개, 하위 10개 데이터에 대한 정보와 소프트 보팅 결과를 화면에 출력.
    inspect_result = inspect_ensemble(test_df, outputs_dict)
    conf_matrix_norm = show_confusion_matrix(CFG, id, outputs_dict, stage='test', ensemble=True)
    # 모든 예측(개별 모델의 예측 + 소프트 보팅의 예측)의 개별 정확도를 화면에 출력.
    indv_model_acc = compare_preds(outputs_list, outputs_dict)
    # 모든 예측(개별 모델의 예측 + 소프트 보팅의 예측)을 대상으로 계산한 상관관계를 화면에 출력.
    corr_data = show_corr(id, outputs_list, outputs_dict)
    ensemble_summary = {'inspect_result': inspect_result,
                        'conf_matrix_norm': conf_matrix_norm,
                        'indv_model_acc': indv_model_acc,
                        'corr_data': corr_data}
    output_path = CFG.output_path + 'test'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path + f'/ensemble_summary_{id}', 'wb') as f:
        pickle.dump(ensemble_summary, f)

    wandb.finish(quiet=True)

if __name__ == '__main__':
  main(args)
