# reference: https://www.kaggle.com/code/debarshichanda/pytorch-feedback-deberta-v3-baseline 바탕으로 작성

import os
import sys
import gc
import wandb
import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm.notebook import tqdm
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, ReduceLROnPlateau, StepLR
from adamp import AdamP
from torchmetrics.functional import accuracy
from transformers import get_linear_schedule_with_warmup

sys.path.append('cd /content/drive/MyDrive/프로젝트/politic_value_relationship/test3/files')
from utils import get_metrics, log_metrics

class Trainer:
    def __init__(self, config, model, dataloaders, is_pm=False):
        self.config = config
        self.model = model.to(config.device)
        self.vocab = config.vocab
        if isinstance(dataloaders, tuple):
            self.train_loader, self.valid_loader = dataloaders
            self.use_optimizer = True
        else:
            self.valid_loader = dataloaders
            self.use_optimizer = False
        self.is_pm = is_pm
        self.train_loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        if self.use_optimizer:
            self.optimizer, self.scheduler = self.configure_optimizers() 
        self.attention_dict = {'loss': np.empty((0,)), 
                               'output': np.empty((0, self.config.output_dim)), 
                               'label': np.empty((0,), dtype='int'), 
                               'pred': np.empty((0,), dtype='int'),
                               'input_ids': np.empty((0, self.config.max_len), dtype='int'), 
                               'attn': np.empty((0, 12, self.config.max_len, self.config.max_len))}

    def train_one_epoch(self):
        """
        Trains the model for 1 epoch
        """
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        pred_list, target_list = [], []
        total_loss = 0

        for step, data_dict in pbar:
            with autocast(enabled=True):
                model_outputs = self.model(data_dict)
                outputs = model_outputs['output']
                preds = outputs.argmax(1)
                targets = data_dict['label']

                loss = self.train_loss_fn(outputs, targets)
                loss_itm = loss.item()
                total_loss += loss_itm
                train_accuracy = accuracy(preds, targets, task="multiclass", num_classes=4)

                pbar.set_description('train_accuracy: {:.2f}'.format(train_accuracy))

            if self.config.accumulation_steps:
                loss /= self.config.accumulation_steps
                self.config.scaler.scale(loss).backward()
                if self.config.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                if (step+1) % self.config.accumulation_steps == 0:
                    self.config.scaler.step(self.optimizer)
                    self.config.scaler.update()
                    self.optimizer.zero_grad()
                    if not self.config.lr_scheduler == 'red':
                        self.scheduler.step()
                    self.model.zero_grad()
            else:
                self.config.scaler.scale(loss).backward()
                if self.config.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.config.scaler.step(self.optimizer)
                self.config.scaler.update()
                self.optimizer.zero_grad()
                if not self.config.lr_scheduler == 'red':
                        self.scheduler.step()
                self.model.zero_grad()
                        
            wandb.log(
                {'train_batch_loss': loss_itm}
            )

            target_list.extend(targets.cpu().detach().numpy().tolist())
            pred_list.extend(preds.cpu().detach().numpy().tolist())

        epoch_avg_loss = round(total_loss/ len(self.train_loader.dataset), 2)
        metrics = get_metrics(target_list, pred_list)
        log_metrics(*metrics, stage='train')
        print(f'[train_epoch_result] => train_epoch_avg_loss: {epoch_avg_loss} | '\
                f'train_epoch_accuracy: {metrics[0]} | '\
                f'train_epoch_precision: {metrics[1]} | '\
                f'train_epoch_recall: {metrics[2]} | '\
                f'train_epoch_f1_score: {metrics[3]}')

        del target_list, pred_list
        gc.collect()
        torch.cuda.empty_cache()
    
    @torch.no_grad()
    def valid_one_epoch(self, stage='valid'):
        """
        Validates the model for 1 epoch
        """
        self.model.eval()
        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader)) 
        outputs_dict = {'output': [], 'pred': [], 'target': [], 'loss': []}

        total_loss = 0

        for idx, data_dict in pbar:
            model_outputs = self.model(data_dict, stage=stage)
            outputs = model_outputs['output']
            preds = outputs.argmax(1)
            
            input_ids = data_dict['input_ids']
            if self.is_pm:
                attn_probs = model_outputs['attn_prob']

            targets = data_dict['label']
            losses = self.loss_fn(outputs, targets)
            loss = torch.mean(losses)
            loss_itm = loss.item()
            total_loss += loss_itm
            valid_accuracy = accuracy(preds, targets, task="multiclass", num_classes=4)

            pbar.set_description(f'{stage}_accuracy: {valid_accuracy:.2f}')
    
            wandb.log(
                {f'{stage}_batch_loss': loss_itm}
            )
            
            outputs_dict['output'].extend(outputs.cpu().detach().numpy().tolist())
            outputs_dict['pred'].extend(preds.cpu().detach().numpy().tolist())
            outputs_dict['target'].extend(targets.cpu().detach().numpy().tolist())
            outputs_dict['loss'].extend(losses.cpu().detach().numpy().tolist())
            
            self.attention_dict['output'] = np.concatenate([self.attention_dict['output'], outputs.cpu().detach().numpy()], axis=0)
            self.attention_dict['pred'] = np.concatenate([self.attention_dict['pred'], preds.cpu().detach().numpy()], axis=0)
            self.attention_dict['label'] = np.concatenate([self.attention_dict['label'], targets.cpu().detach().numpy()], axis=0)
            self.attention_dict['loss'] = np.concatenate([self.attention_dict['loss'], losses.cpu().detach().numpy()], axis=0)
            
            # attention 시각화를 위해 필요 정보를 수집
            if self.is_pm:
                self.attention_dict['input_ids'] = np.concatenate([self.attention_dict['input_ids'], input_ids.cpu().detach().numpy()], axis=0)
                self.attention_dict['attn'] = np.concatenate([self.attention_dict['attn'], attn_probs.cpu().detach().numpy()], axis=0)
            
            # GPU 메모리에 부담이 되지 않게 attention 시각화 관련 정보를 Loss를 기준으로 필터링
            if self.attention_dict['output'].shape[0] >= 32:
              self.filter_attention_dict()
        
        epoch_avg_loss = round(total_loss/ len(self.valid_loader.dataset), 4)
        outputs_dict['epoch_avg_loss'] = epoch_avg_loss
        
        metrics = get_metrics(outputs_dict['target'], outputs_dict['pred'])
        log_metrics(*metrics, stage='valid')
        print(f'[valid_epoch_result] => valid_epoch_avg_loss: {epoch_avg_loss} | '\
              f'valid_epoch_accuracy: {metrics[0]} | '\
                f'valid_epoch_precision: {metrics[1]} | '\
                f'valid_epoch_recall: {metrics[2]} | '\
                f'valid_epoch_f1_score: {metrics[3]}')
        
        torch.cuda.empty_cache()
                
        return outputs_dict
    
    def fit(self):
        """
        Low-effort alternative for doing the complete training and validation process
        """
        best_loss = int(1e+7)
        early_stopping_count = 0

        for epx in range(self.config.epochs):
            print(f"{'='*20} Epoch: {epx+1} / {self.config.epochs} {'='*20}")

            self.train_one_epoch()
            outputs_dict = self.valid_one_epoch()
            epoch_avg_loss = outputs_dict['epoch_avg_loss']
            if self.config.lr_scheduler == 'red':
                self.scheduler.step(epoch_avg_loss)
            
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                data = {'epoch': epx+1,
                        'state_dict': self.model.state_dict()}
                if not self.is_pm:
                    data['vocab'] = self.vocab
                self.save_model(data)
                print(f"Saved model with val_loss: {best_loss:.4f}")
                early_stopping_count = 0
            else:
                early_stopping_count += 1
                if early_stopping_count == self.config.patience:
                    print(f'Early stopping after {epx+1} epochs')
                    break
                
        return outputs_dict
        
    def test(self):
        outputs_dict = self.valid_one_epoch(stage='valid') # wandb에서 valid와 test 성능을 비교하기 위해 테스트 때도 stage='valid' 사용
        return outputs_dict

    def predict(self):
        self.model.eval()
        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc='processing output') 
        outputs_dict = {'output': [], 'pred': []} # 

        for idx, data_dict in pbar:
            model_outputs = self.model(data_dict, stage='predict')
            outputs = model_outputs['output']
            preds = outputs.argmax(1)
            
            input_ids = data_dict['input_ids']
            if self.is_pm:
                attn_probs = model_outputs['attn_prob']

            outputs_dict['output'].extend(outputs.cpu().detach().numpy().tolist())
            outputs_dict['pred'].extend(preds.cpu().detach().numpy().tolist())

            self.attention_dict['output'] = np.concatenate([self.attention_dict['output'], outputs.cpu().detach().numpy()], axis=0)
            self.attention_dict['pred'] = np.concatenate([self.attention_dict['pred'], preds.cpu().detach().numpy()], axis=0)
            self.attention_dict['input_ids'] = np.concatenate([self.attention_dict['input_ids'], input_ids.cpu().detach().numpy()], axis=0)
            if self.is_pm:
                self.attention_dict['attn'] = np.concatenate([self.attention_dict['attn'], attn_probs.cpu().detach().numpy()], axis=0)
            
            if self.attention_dict['output'].shape[0] >= 40:
              self.filter_attention_dict()        

        return outputs_dict
    
    def configure_optimizers(self):
        lr = self.config.pm_lr if self.is_pm else self.config.lr
        if self.config.optim == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif self.config.optim == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif self.config.optim == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        elif self.config.optim == 'AdamP':
            optimizer = AdamP(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError('Only Adam, RMSprop, AdamW and AdamP are Supported!')

        if self.config.lr_scheduler == 'gls':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warm_steps, num_training_steps=len(self.train_loader)*self.config.epochs)
        elif self.config.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.config.T_0, T_mult=self.config.T_mult, eta_min=self.config.η_min)
        elif self.config.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=self.config.gamma)
        elif self.config.lr_scheduler == 'red':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        elif self.config.lr_scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return optimizer, scheduler

    def save_model(self, data, verbose=False):
        """
        Saves the model at the provided destination
        """            
        path = self.config.model_path
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except:
            print("Errors encountered while making the output directory")

        # P: pretrained, D: deeplearning, F: fold
        if self.is_pm:
            if self.config.fusion:
                name = f'P_fusion_{self.model.__class__.__name__}_F{self.config.fold}.pt'
            else:
                name = f'P_{self.model.__class__.__name__}_F{self.config.fold}.pt'
        else:
            if self.config.fusion:
                name = f'D_fusion_{self.model.__class__.__name__}_F{self.config.fold}.pt'
            else:
                name = f'D_{self.model.__class__.__name__}_F{self.config.fold}.pt'
        
        torch.save(data, os.path.join(path, name))
        if verbose:
            print(f"Model Saved at: {os.path.join(path, name)}")
    
    def filter_attention_dict(self, num_top=20, num_bottom=20):
      top_idx = np.argsort(self.attention_dict['loss'])[-num_top:]
      bottom_idx = np.argsort(self.attention_dict['loss'])[:num_bottom]
      
      for key, value in self.attention_dict.items():
        if self.is_pm:
            self.attention_dict[key] = np.concatenate((value[top_idx],  value[bottom_idx]), axis=0)
        else:
            if key not in ['input_ids', 'attn']:
                self.attention_dict[key] = np.concatenate((value[top_idx],  value[bottom_idx]), axis=0)
      
      del top_idx, bottom_idx
      gc.collect()
