"""
전통적인 머신러닝 모델을 시험하는 라이브러리.

시험 데이터를 불러온 후 훈련된 모델 중 하나를 선택하고 ml_trainer 모듈을 이용해 모델을 시험합니다.
모델 성능 지표는 wandb 라이브러리를 이용해 기록합니다.
모델의 반환값 모두를 파일에 저장하고, 주요 반환값을 화면에 출력하며, 혼동 행렬도 시각화합니다. 
"""


import os 
import sys
import random
import glob
import wandb
import pandas as pd
import numpy as np
from konlpy.tag import Mecab

sys.path.append('cd /content/drive/MyDrive/프로젝트/politic_value_relationship/test3/files')
from ml_trainer import ML_Trainer
from utils import initialize_wandb, summarize_result, show_confusion_matrix

class CFG:
    seed = 7
    fusion = False # False: <온라인 커뮤니티 데이터>로 학습한 모델을 사용/ True : <온라인 커뮤니티 데이터 + 네이버 댓글 데이터>로 학습한 모델을 사용.
    path_idx = 2 # 모델 경로 리스트에서 특정 모델 경로를 선택하는 인덱스.
    model = None
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
        'path_idx': CFG.path_idx,
        'arch': 'ML',
        'data': None
        }

def ml_test(model_path, test_df, id):
    trainer = ML_Trainer(CFG, test_df, stage='test')
    trainer.load(model_path)

    outputs_dict = trainer.test()
    # summarize_result 함수에서 사용하기 위해 입력 텍스트를 수집.
    outputs_dict['content'] = test_df.document.values

    # utils 모듈의 summarize_result 함수를 이용해 모델의 반환값을 파일에 모두 저장한 후 loss를 기준으로 상위 10개, 하위 10개의 모델 예측과 라벨을 화면에 출력.
    summarize_result(CFG, id, outputs_dict, 'test')
    # utils 모듈의 show_confusion_matrix 함수를 이용해 혼동 행렬을 화면에 출력.
    show_confusion_matrix(CFG, id, outputs_dict, 'test')
    
    wandb.finish(quiet=True)

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

    # 훈련된 모델 중 하나를 선택하여 테스트를 진행하고, 모델의 주요 예측 결과와 혼동 행렬을 화면에 출력.
    if CFG.model == 'LR':
        model_path_list = glob.glob(os.path.join(CFG.model_path, '*LR*'))
        selected_model_path = model_path_list[CFG.path_idx]
    elif CFG.model == 'NB':
        model_path_list = glob.glob(os.path.join(CFG.model_path, '*NB*'))
        selected_model_path = model_path_list[CFG.path_idx]
    elif CFG.model == 'LGBM':
        model_path_list = glob.glob(os.path.join(CFG.model_path, '*LGBM*'))
        selected_model_path = model_path_list[CFG.path_idx]
    print('model path_list -> ', model_path_list)
    print('selected_model path -> ', selected_model_path)

    ml_test(selected_model_path, test_df, id)

if __name__ == '__main__':
    main(args)

