import os
import sys
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append('cd /content/drive/MyDrive/프로젝트/politic_value_relationship/test3/files')
from visualizer import int2str

def initialize_wandb(config, args, stage):
    if isinstance(config, tuple):
        rnn_config, cnn_config = config
        if args.model == 'rnn':
            wandb_config = rnn_config
        else:
            wandb_config = cnn_config
    else:
        wandb_config = config
    
    wandb_config['model'] = args.model
    wandb_config['file'] = args.file_name
    model = args.model
    time = log_time()
    if stage == 'train':
        wandb_config['fold'] = args.fold
        fold = args.fold
        id =  f'fusion_{model}_{time}_F{fold}' if args.fusion else f'{model}_{time}_F{fold}'
    else:
        if args.model == 'multi-model':
            id =  f'fusion_{model}_{time}' if args.fusion else f'{model}_{time}'
        else:
            wandb_config['fold'] = args.path_idx
            path_idx = args.path_idx
            id =  f'fusion_{model}_{time}_P{path_idx}' if args.fusion else f'{model}_{time}_P{path_idx}'

    wandb.init(
        project='PVR-TEST3',
        config= wandb_config,
        group= 'fusion_model' if args.fusion else 'base_model',
        job_type=stage,
        tags = ['community_data + naver data'] if args.fusion else ['community_data'],
        name = id
        )
    
    return id

def log_time():
  import datetime
  import pytz
  now_utc = datetime.datetime.utcnow()
  kr_tz = pytz.timezone('Asia/Seoul')
  now_kr = now_utc.replace(tzinfo=pytz.utc).astimezone(kr_tz)
  year = now_kr.year
  month = now_kr.month
  day = now_kr.day
  hour = now_kr.hour
  minute = now_kr.minute
  now_time = f'{year}.{month}.{day} {hour}:{minute}'
  return now_time

def get_metrics(targets, preds):
  accuracy = round(accuracy_score(targets, preds), 2)
  precision = round(precision_score(targets, preds, average='weighted', zero_division=True), 2)
  recall = round(recall_score(targets, preds, average='weighted'), 2)
  f1_socre = round(f1_score(targets, preds, average='weighted'), 2)
  return accuracy, precision, recall, f1_socre

def log_metrics(*args, stage='train'):
  accuracy, precision, recall, f1_score = args
  wandb.log(
    {f'{stage}_epoch_accuracy': accuracy,
     f'{stage}_epoch_precision': precision,
     f'{stage}_epoch_recall': recall,
     f'{stage}_epoch_f1_score': f1_score}
    )
  
def summarize_result(config, model_id, outputs_dict, stage):
    if 'target' in outputs_dict.keys():
        df = pd.DataFrame({'loss': outputs_dict['loss'],
                           'pred': outputs_dict['pred'],
                           'label': outputs_dict['target'],
                           'content': outputs_dict['content']})
        all_df = df.sort_values('loss', ascending=False)
        output_path = config.output_path + stage
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        all_df[['pred', 'label']] = all_df[['pred', 'label']].applymap(int2str)
        all_df.to_csv(output_path + f'/{stage}_all_preds_{model_id}.csv', index=False)
        correct_df = df[df['pred'] == df['label']].sort_values('loss').head(10)
        wrong_df = df[df['pred'] != df['label']].sort_values('loss', ascending=False).head(10)
        correct_df[['pred', 'label']] = correct_df[['pred', 'label']].applymap(int2str)
        wrong_df[['pred', 'label']] = wrong_df[['pred', 'label']].applymap(int2str)

        print(f'|| 모델: {model_id} ||')
        print('='*50 + '정답 예측 & loss 낮음 TOP10' + '='*50)
        print('수정 내용 반영')
        display(HTML(correct_df.to_html())) 
        print()
        print(f'|| 모델: {model_id} ||')
        print('='*50 + '오답 예측 & loss 높음 TOP10' + '='*50)
        display(HTML(wrong_df.to_html()))
    else:
        df = pd.DataFrame({'pred': outputs_dict['pred'], 'content': outputs_dict['content']})
        output_path = config.output_path + stage
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        df.to_csv(config.output_path + f'/{stage}_result_{model_id}.csv', index=False)
        print(f'|| 모델: {model_id} ||')
        display(HTML(df.to_html()))
  
def show_confusion_matrix(config, model_id, outputs_dict, stage, ensemble=False):
        conf_matrix = confusion_matrix(outputs_dict['target'], outputs_dict['pred'])
        labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

        # normalize confusion matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        # plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix_norm, annot=True, annot_kws={"size": 17}, cmap='Blues', fmt='.2f', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title('Confusion Matrix', fontsize=20, pad=20)
        ax.set_xlabel('Prediction', fontsize=16, labelpad=20)
        ax.set_ylabel('Label', fontsize=16, labelpad=20)
        ax.tick_params(axis='x', labelsize=14, pad=10)
        ax.tick_params(axis='y', labelsize=14, pad=10)
        plt.tight_layout()
        output_path = config.output_path + stage
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(output_path + f'/{stage}_cmat_{model_id}.png')
        plt.show()
        
        if ensemble:
            return conf_matrix_norm
