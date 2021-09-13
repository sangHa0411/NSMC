import collections
import re
from enum import IntEnum
import random

class Token(IntEnum) :
    PAD = 0
    UNK = 1

def preprocess(sen) :
    sen = re.sub('[0-9]+,*[0-9]*', 'NUM ', sen) # 숫자는 NUM 토큰으로 으로 치환
    sen = re.sub('[^A-Z가-힣!?.,\']', ' ' , sen) # 특수문자 제거

    # 중복 제거 [? ! . ,]
    sen = re.sub('[.]{2,}' , '.' , sen)
    sen = re.sub('[,]{2,}' , ',' , sen)
    sen = re.sub('[?]{2,}' , '?' , sen)
    sen = re.sub('[!]{2,}' , '!' , sen)

    # 중복 공백 하나로 만들기
    sen = re.sub(' {2,}', ' ', sen)
    return sen

class Preprocessor :
    def __init__(self, raw_text, tokenize_fn, th=5) :
        self.tok_text = [tokenize_fn(preprocess(text)) for text in raw_text]
        self.tokenize_fn = tokenize_fn
        self.th = th
        
        self.build_data()

    def build_data(self) :
        counter = collections.Counter()
        for tok_list in self.tok_text :
            counter.update(tok_list)
        counter = dict(counter)
        
        valid_tok = []
        for tok, count in counter.items() :
            if (count >= self.th) :
                valid_tok.append(tok)  
        random.shuffle(valid_tok)     
        tok_sel = ['PAD', 'UNK'] + valid_tok
        
        self.word2idx = dict(zip(tok_sel, range(len(tok_sel))))
    
    def set_data(self, data) :
        idx_list = list(data.values())
        tok_list = list(data.keys())
        
        word2idx = {}
        for i in range(len(data)) :
            word2idx[tok_list[i]] = idx_list[i]
            
        self.word2idx = word2idx
    
    def get_data(self) :
        return self.word2idx

    # encode token list
    def encode_sen(self, sen) :
        idx_list = []
        for tok in sen :
            idx = self.word2idx[tok] if tok in self.word2idx else Token.UNK
            idx_list.append(idx)
        return idx_list
    
    # encode whole data
    def encode(self) :
        idx_data = [self.encode_sen(sen) for sen in self.tok_text]
        return idx_data