import os
import re
import json
from dataset import Token
from tqdm import tqdm
import sentencepiece as spm

spm_templates= '--input={} \
--pad_id={} \
--bos_id={} \
--eos_id={} \
--unk_id={} \
--model_prefix={} \
--vocab_size={} \
--character_coverage={} \
--model_type={}'

def read_data(file_path) :
    with open(file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data         
              
def get_data(json_data) :
    text_data = []
    doc_data = json_data['document']
    for doc in doc_data :
        data_list = doc['utterance']
        dialogue = [data['form'] for data in data_list if len(data['form']) >= 5]
        text_data.extend(dialogue)
    return text_data

def load_data(dir_path) :
    data_list = os.listdir(dir_path)
    json_data = []
    for data in data_list :
        if data.endswith('.json') :
            file_path = os.path.join(dir_path, data)
            
            try :
                json_file = read_data(file_path)
                json_data.append(json_file)
            except UnicodeDecodeError :
                continue

    text_data = []
    for json_file in tqdm(json_data) :
        text_list = get_data(json_file)
        text_data.extend(text_list)
    return text_data
 
def preprocess_kor(sen) :
    sen = re.sub('[^가-힣0-9 ~\',.!?]' , '', sen)
    sen = re.sub(' {2,}' , ' ' , sen)
    return sen

def write_data(text_list, text_path, preprocess) :
    with open(text_path, 'w') as f :
        for sen in text_list :
            sen = preprocess(sen)
            f.write(sen + '\n')

def train_spm(dir_path, data, model, vocab_size) :
    text_path = os.path.join(dir_path, data)
    spm_cmd = spm_templates.format(text_path, 
            Token.PAD,
            Token.SOS, 
            Token.EOS, 
            Token.UNK, 
            os.path.join(dir_path, model), 
            vocab_size, 
            1.0, 
            'unigram')
    spm.SentencePieceTrainer.Train(spm_cmd)

def get_spm(dir_path, model) :
    model_path = os.path.join(dir_path, model)
    if os.path.exists(model_path) :
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        sp.SetEncodeExtraOptions('bos:eos')
        return sp
    else:
        raise FileNotFoundError
