import numpy as np
import pandas as pd
from IPython.display import HTML, display
from transformers import AutoTokenizer

class CFG:
    pm_model_name = 'beomi/kcbert-base'
    pm_tokenizer = AutoTokenizer.from_pretrained(pm_model_name)

class Visualizer:
    def __init__(self, config, model_path, model_id, data):
        attn = data.get('attn')
        self.has_attn = bool(np.any(data['attn'])) if attn is not None else False
        self.config = config
        self.model_id = model_id
        self.data = data
    
    def show_attention(self):
        assert self.has_attn, '어텐션 시각화 불가능한 모델과 결과'
        reference_idx = np.argsort(self.data['loss'])
        loss_list = self.data['loss'][reference_idx]
        pred_list =  self.data['pred'][reference_idx]
        label_list =  self.data['label'][reference_idx]
        input_ids_list =  self.data['input_ids'][reference_idx].tolist()
        attn_prob_list =  [self.data['attn'][idx]for idx in reference_idx]

        df = pd.DataFrame({'loss': loss_list, 'pred': pred_list, 'label': label_list})
        loss_mean = df['loss'].mean()
        good_pred_idx = df[df['pred'] == df['label']].sort_values('loss').index[:10]
        bad_pred_idx = df[df['pred'] != df['label']].sort_values('loss').index[-10:]

        for i, idx in enumerate(list(good_pred_idx) + list(bad_pred_idx)):
            if i <= len(good_pred_idx):
                print('<정답 예측 & loss 낮음>')
            else:
                print('<오답 예측 & loss 높음>')
            loss, pred, label= df.iloc[idx].values
            pred = int2str(pred)
            label = int2str(label)
            loss_ratio = calc_loss_ratio(loss, loss_mean)
            token_ids = input_ids_list[idx]
            attn_prob = attn_prob_list[idx]
            print(f'정답: {label}, 예측: {pred}, loss: {loss_ratio}')
            html_output = mk_html(token_ids, attn_prob)  
            display(HTML(html_output))
            print('-'*200)

def int2str(int_output):
        if int_output == 0:
            str_output = "국힘갤 부류"
        elif int_output == 1:
            str_output = "뽐뿌 부류"
        elif int_output == 2:
            str_output = "펨코 부류"
        elif int_output == 3:
            str_output = "루리웹 부류"
        return str_output
    
def calc_loss_ratio(loss, loss_mean):
    ratio = round(((loss / loss_mean) - 1) * 100, 2)
    if ratio < 0:
        output = f'loss 평균 대비 {abs(ratio)}%만큼 낮음'
    else:
        output = f'loss 평균 대비 {ratio}%만큼 높음'
    return output 

def highlight(word, attn):
    # "Attention 값이 크면 문자 배경을 진한 빨간색으로 하는 html을 출력하는 함수"
    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn))
        )
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)

def mk_html(sentence, normlized_weights):
    stop_cnt = sentence.count(3) # vocab의 idx가 3인 토큰: '[SEP]'
    tokenizer = CFG.pm_tokenizer

    # 표시용의 HTML을 작성한다
    html = ''

    # Self-Attention의 가중치를 가시화. Multi-Head가 12개이므로, 12종류의 attention이 존재
    for i in range(12):

        # index의 Attention을 추출하고 규격화
        # 0번째 단어 [CLS]의, i번째의 Multi-Head Attention를 꺼냄
        # index는 미니 배치의 몇 번째 데이터인지를 나타냄
        attens = normlized_weights[i, 0, :]
        attens /= attens.max()

        html += '[BERT의 Attention을 시각화_' + str(i+1) + ']<br>'
        cnt = 0
        for word, attn in zip(sentence, attens):

            # 단어가 [SEP]인 경우 문장의 끝이므로 break
            if tokenizer.convert_ids_to_tokens(word) == "[SEP]":
                cnt += 1
            if cnt == stop_cnt:
                break

            # 함수 highlight로 색을 칠하고, 함수 tokenizer_bert.convert_ids_to_tokens로 ID를 단어로 되돌림
            html += highlight(tokenizer.convert_ids_to_tokens(
                word), attn)
        html += "<br><br>"

    # 12종류의 Attention의 평균을 구한다. 최대치로 규격화
    for i in range(12):
        attens += normlized_weights[i, 0, :]
    attens /= attens.max()

    html += '[BERT의 Attention을 시각화_ALL]<br>'
    cnt = 0
    for word, attn in zip(sentence, attens):
        # 단어가 [SEP]일 경우는 문장의 끝이므로 break
        if tokenizer.convert_ids_to_tokens(word) == "[SEP]":
            cnt += 1
        if cnt == stop_cnt:
            break

        # 함수 highlight로 색을 칠하고, 함수 tokenizer_bert.convert_ids_to_tokens로 ID를 단어로 되돌림
        html += highlight(tokenizer.convert_ids_to_tokens(
            word), attn)
    html += "<br><br>"
    return html