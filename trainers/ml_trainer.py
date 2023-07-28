"""
전통적인 머신러닝 모델의 훈련, 검증, 시험을 담당하고 하이퍼파라미터 튜닝도 수행하는 라이브러리.

입력 데이터로 TF-IDF 행렬를 만든 후 Logistic Regression, Naive Bayes, LightGBM 모델 중 하나로 훈련, 시험, 검증 작업을 수행합니다.

사용 가능 클래스
    ML_Trainer: 설정값으로 지정한 머신러닝 모델로 훈련, 검증, 시험 작업을 진행하고, 하이퍼파라미터 튜닝 수행합니다.
"""


import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss

from utils import get_metrics
from utils import log_metrics


class ML_Trainer:
    def __init__(self, config, data, model=None, stage='train'):
        self.config = config
        self.model = model
        if isinstance(data, tuple):
            self.train_df, self.valid_df = data
        elif isinstance(data, pd.DataFrame):
            self.valid_df = data
        self.tokenizer = config.tokenizer
        self.vectorizer = None 
        self.transformer = None 
        self.tfidf_matrix = None

    def make_embedding(self, ngram_range=(1, 2), n_features=2**20):
        start = end = time.time()
        # TfidfVectorizer를 사용하기에는 TF-IDF 행렬의 열을 구성하는 토큰 갯수가 너무 많을 수 있으므로 HashingVectorizer를 사용.
        self.vectorizer = HashingVectorizer(input="content", tokenizer=self.tokenizer.morphs, ngram_range=ngram_range, n_features=n_features, alternate_sign=False)
        train_input_counts = self.vectorizer.fit_transform(self.train_df.document.values)
        self.transformer = TfidfTransformer(use_idf=True,).fit(train_input_counts)
        self.tfidf_matrix = self.transformer.transform(train_input_counts)
        
        print('TF-IDF Matrix 생성에 걸린 시간: ', round(time.time() - end, 2))

    def tune_hyperparm(self, params, how='random'):
        start = end = time.time()
        if how == 'random':
            search = RandomizedSearchCV(self.model, param_distributions=params, n_iter=self.config.n_iter, cv=self.config.cv, n_jobs=-1, verbose=1, scoring='accuracy', refit=True) # random_state/ n_iter: 랜덤하게 조합을 탐색할 횟수
        else:
            search = GridSearchCV(self.model, param_grid=params, cv=self.config.cv, scoring='accuracy', n_jobs=-1, verbose=1)
        
        # LightGBM은 fit 메소드에 사용하는 인자가 일부 다르므로 별도로 처리.
        if self.config.use_gbm:
            inputs_counts = self.vectorizer.transform(self.valid_df.document.values)
            inputs_tfidf = self.transformer.transform(inputs_counts)
            search.fit(self.tfidf_matrix, self.train_df.label.values,
                       early_stopping_rounds=50, eval_metric='logloss', eval_set=[(inputs_tfidf, self.valid_df.label.values)])
        else:
            search.fit(self.tfidf_matrix, self.train_df.label.values)

        print('파라미터 튜닝에 걸린 시간: ', round(time.time() - end, 2))
        print('최적 파라미터: ', search.best_params_)
        print('최고 정확도: ', search.best_score_)

        self.train_accuracy = round(search.best_score_, 2)
        self.model = search.best_estimator_

    def fit(self, params=None, ngram_range=(1, 2), n_features=2**20, how='random', have_embedding=False):
        if not have_embedding:
            self.make_embedding(ngram_range=ngram_range, n_features=n_features)
        
        if params:
            self.tune_hyperparm(params=params, how=how)
        else:
            print(self.train_df.label.values.shape)
            self.model.fit(self.tfidf_matrix, self.train_df.label.values)
            target = self.train_df.label.values
            pred_proba = self.model.predict_proba(self.tfidf_matrix)
            pred = pred_proba.argmax(axis=1)

            epoch_avg_loss = log_loss(target, pred_proba)
            metrics = get_metrics(target, pred)
            log_metrics(*metrics, stage='train')
            print(f'[train_epoch_result] => train_epoch_avg_loss: {epoch_avg_loss} | '\
                    f'train_epoch_accuracy: {metrics[0]} | '\
                    f'train_epoch_precision: {metrics[1]} | '\
                    f'train_epoch_recall: {metrics[2]} | '\
                    f'train_epoch_f1_score: {metrics[3]}')
        
        output_dict = self.forward()

        self.save()
        return output_dict

    def save(self):
        data = {'model': self.model,
                'vectorizer': self.vectorizer,
                'transformer': self.transformer}
        path = self.config.model_path
        if self.config.fusion:
            name = f'M_fusion_{self.config.model}_f{self.config.fold}.joblib'
        else:
            name = f'M_{self.config.model}_f{self.config.fold}.joblib'
        joblib.dump(data, os.path.join(path, name))

    def load(self, path, name=None, verbose=False):
        if name is None:
            data = joblib.load(path)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.transformer = data['transformer']
            return
        object = joblib.load(os.path.join(path, name))
        return object
        
    def forward(self):
        inputs_counts = self.vectorizer.transform(self.valid_df.document.values)
        inputs_tfidf = self.transformer.transform(inputs_counts)
        pred_proba = self.model.predict_proba(inputs_tfidf)
        pred = pred_proba.argmax(axis=1)
        target = self.valid_df.label.values
        
        epoch_avg_loss = log_loss(target, pred_proba, labels=np.unique(target))
        metrics = get_metrics(target, pred)
        log_metrics(*metrics, stage='valid')
        print(f'[valid_epoch_result] => valid_epoch_avg_loss: {epoch_avg_loss} | '\
                f'valid_epoch_accuracy: {metrics[0]} | '\
                  f'valid_epoch_precision: {metrics[1]} | '\
                  f'valid_epoch_recall: {metrics[2]} | '\
                  f'valid_epoch_f1_score: {metrics[3]}')

        loss = [log_loss([target[i]], [pred_proba[i]], labels=np.unique(target)) for i in range(len(target))]
        output_dict = {'loss': loss, 'output': pred_proba,'pred': pred, 'target': target}
        return output_dict

    def test(self):
        output_dict = self.forward()
        return output_dict

    def predict(self):
        inputs_counts = self.vectorizer.transform(self.valid_df.document.values)
        inputs_tfidf = self.transformer.transform(inputs_counts)
        pred_proba = self.model.predict_proba(inputs_tfidf)
        pred = pred_proba.argmax(axis=1)
        output_dict = {'output': pred_proba, 'pred': pred}        
        return output_dict
