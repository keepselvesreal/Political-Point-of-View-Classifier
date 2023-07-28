"""
사전훈련 모델을 시험하는 라이브러리.

시험 데이터를 불러온 후 훈련된 모델 중 하나를 선택하고 pm_datamodule, trainer 라이브러리를 이용해 모델을 시험합니다.
모델 성능 지표는 wandb 라이브러리를 이용해 기록합니다.
모델의 반환값 모두를 파일에 저장하고, 주요 반환값을 화면에 출력하며, 혼동 행렬도 시각화합니다. 
"""


import os
import sys
import random
import glob
import pickle
import wandb
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.append('cd /content/drive/MyDrive/프로젝트/politic_value_relationship/test3/files')
from pm_datamodule import PM_DataModule
from trainer import Trainer
from pm_model import Kcbert
from visualizer import Visualizer
from utils import initialize_wandb, summarize_result, show_confusion_matrix

class CFG:
    seed = 7
    path_idx = 2 # 모델 경로 리스트에서 특정 모델 경로를 선택하는 인덱스.
    fusion = True #  False: <온라인 커뮤니티 데이터>로 학습한 모델을 사용/ True : <온라인 커뮤니티 데이터 + 네이버 댓글 데이터>로 학습한 모델을 사용.
    
    model = 'Kcbert'
    pm_batch_size = 16
    vocab = None
    load_data = False
    output_dim = 4
    max_len = 300
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    pm_model_name = 'beomi/kcbert-base'
    pm_tokenizer = AutoTokenizer.from_pretrained(pm_model_name)

    csv_path = '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/test.csv'
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
    'path_idx': CFG.path_idx,
    'arch': 'PM',
    'data': None
    }

def pm_test(model_path, test_df, id, ensemble=False):
    data = torch.load(model_path) 

    model = Kcbert(CFG)
    state_dict = data['state_dict']
    model.load_state_dict(state_dict)

    datamodule = PM_DataModule(CFG, test_df)
    test_dataloader = datamodule.build_test_dataloader()
    
    trainer = Trainer(CFG, model, test_dataloader, is_pm=True)
    outputs_dict = trainer.test()
    # # summarize_result 함수와 visualizer 객체에서 사용하기 위해 입력 텍스트를 수집.
    outputs_dict['content'] = test_df.document.values

    print()
    # utils 모듈의 summarize_result 함수를 이용해 모델의 반환값을 파일에 모두 저장한 후 loss를 기준으로 상위 10개, 하위 10개의 모델 예측과 라벨을 화면에 출력.
    summarize_result(CFG, id, outputs_dict, 'test')
    print()
    # utils 모듈의 show_confusion_matrix 함수를 이용해 혼동 행렬을 화면에 출력.
    show_confusion_matrix(CFG, id, outputs_dict, 'test')
    attention_dict = trainer.attention_dict  
    visualizer = Visualizer(CFG, model_path, id, attention_dict)
    print()
    # 입력 텍스트에 모델의 어텐션 점수를 색의 농도로 표현하여 화면에 출력.
    visualizer.show_attention()
    output_path = CFG.output_path + 'train'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path + f'/test_visualizer_{id}', 'wb') as f:
        pickle.dump(visualizer, f)
    if ensemble:
        return model.__class__.__name__, outputs_dict

def main(args):
    CFG.model = args.model
    CFG.path_idx = args.path_idx
    CFG.fusion = args.fusion
    file_name = args.file_name
    CFG.csv_path = f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}' if CFG.fusion else f'/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/data/{file_name}'
    CFG.model_path = '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/fusion_models/' if CFG.fusion else '/content/drive/MyDrive/프로젝트/politic_value_relationship/test3/base_models/'

    # utils 모듈의 initialize_wandb 함수를 이용해 wandb 설정값을 입력하고 기록을 시작. wandb 기록 확인에 사용할 식별자(str)를 반환.
    id = initialize_wandb(WANDB_CONFIG, args, 'test')
    
    test_df = pd.read_csv(CFG.csv_path)
    print('test_df shape', test_df.shape)
    print()

    model_path_list = glob.glob(os.path.join(CFG.model_path, 'P*'))
    print('model path list -> ', model_path_list)
    selected_model_path = model_path_list[CFG.path_idx]
    print('selected model path-> ', selected_model_path)
    pm_test(selected_model_path, test_df, id)
    wandb.finish(quiet=True)

if __name__ == '__main__':
    main(args)

