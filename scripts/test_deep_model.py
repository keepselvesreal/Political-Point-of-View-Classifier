"""
딥러닝 모델을 시험하는 스크립트.

시험 데이터를 불러온 후 훈련된 모델 중 하나를 선택하고 dl_datamodule, trainer 모듈을 이용해 모델을 시험합니다.
모델 성능 지표는 wandb 라이브러리를 이용해 기록합니다.
모델의 반환값 모두를 파일에 저장하고, 주요 반환값을 화면에 출력하며, 혼동 행렬도 시각화합니다. 
"""


import os
import sys
import random
import glob
import wandb
import numpy as np
import pandas as pd
import torch
from konlpy.tag import Mecab

sys.path.append('cd /content/drive/MyDrive/프로젝트/politic_value_relationship/test3/files')
from dl_datamodule import DL_DataModule
from trainer import Trainer
from dl_model import Rnn, Cnn
from utils import initialize_wandb, summarize_result, show_confusion_matrix

class CFG:
    seed = 7
    path_idx = 2 # 모델 경로 리스트에서 특정 모델 경로를 선택하는 인덱스.
    fusion = False # False: <온라인 커뮤니티 데이터>로 학습한 모델을 사용/ True : <온라인 커뮤니티 데이터 + 네이버 댓글 데이터>로 학습한 모델을 사용.
    
    model = None
    batch_size = 64
    load_data = False
    tokenizer = Mecab()
    tokenizer_type = 'morphs'
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
    num_layers = 5


    window_size = 5
    window_size_list = [2 ,4, 6]
    num_filter = 100
    num_filter_list = [150, 150, 150] 
    use_batch_norm = False
    activation = 'ReLU'

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
    'path_idx': CFG.path_idx,
    'arch': 'DL',
    'data': None,

    'rnn_type': CFG.rnn_type,
    'embedding_dim': CFG.embedding_dim,
    'hidden_dim': CFG.hidden_dim,    
    'num_layers': CFG.num_layers,
    'bidirectional': CFG.bidirectional,
    'dropout': CFG.dropout
}
CNN_CONFIG = {
    'model': CFG.model,
    'path_idx': CFG.path_idx,
    'arch': 'DL',

    'window_size_list': CFG.window_size_list,
    'num_filter_list': CFG.num_filter_list,
    'dropout': CFG.dropout,
    'use_batch_norm': CFG.use_batch_norm
}

def dl_test(model_path, test_df, model, id, ensemble=False):
    data = torch.load(model_path)

    # 저장된 데이터에서 모델 훈련 때 생성한 사전을 가져옴.
    vocab = data['vocab']
    datamodule = DL_DataModule(CFG, test_df)
    # 데이터모듈 객체에 위에서 가져온 사전을 전달.   
    datamodule.vocab = vocab
    test_dataloader = datamodule.build_test_dataloader()

    # 딥러닝 모델의 임베딩 층을 만드는 데 필요한 토큰 개수 정보 저장.
    CFG.vocab_size = len(vocab)
    model = model(CFG)
    state_dict = data['state_dict']
    model.load_state_dict(state_dict)

    trainer = Trainer(CFG, model, test_dataloader, is_pm=False)
    outputs_dict = trainer.test()
    outputs_dict['content'] = test_df.document.values

    # utils 모듈의 summarize_result 함수를 이용해 모델의 반환값을 파일에 모두 저장한 후 loss를 기준으로 상위 10개, 하위 10개의 모델 예측과 라벨을 화면에 출력.
    summarize_result(CFG, id, outputs_dict, 'test')
    # utils 모듈의 show_confusion_matrix 함수를 이용해 혼동 행렬을 화면에 출력.
    show_confusion_matrix(CFG, id, outputs_dict, 'test')
    
    wandb.finish(quiet=True)

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
    id = initialize_wandb((RNN_CONFIG, CNN_CONFIG), args, 'test')

    test_df = pd.read_csv(CFG.csv_path)
    print('test_df shape', test_df.shape)

    # 훈련된 모델 중 하나를 선택하여 테스트를 진행하고, 모델의 주요 예측 결과와 혼동 행렬을 화면에 출력. 
    if CFG.model == 'rnn':
        rnn_model_path_list = glob.glob(os.path.join(CFG.model_path, '*Rnn*'))
        print('RNN model path list -> ', rnn_model_path_list)
        selected_model_path = rnn_model_path_list[CFG.path_idx]
        print('selected model path -> ', selected_model_path)
        dl_test(selected_model_path, test_df, Rnn, id)
    elif CFG.model == 'cnn':
        cnn_model_path_list = glob.glob(os.path.join(CFG.model_path, '*Cnn*'))
        print('CNN model path list-> ', cnn_model_path_list)
        selected_model_path = cnn_model_path_list[CFG.path_idx]
        print('selected model path -> ', selected_model_path)
        dl_test(selected_model_path, test_df, Cnn, id)
        
if __name__ == '__main__':
    main(args)


