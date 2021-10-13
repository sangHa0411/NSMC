import re

class SenPreprocessor :
    def __init__(self, tokenizer) :
        assert hasattr(tokenizer, 'morphs')
        self.filter = re.compile('[^a-zA-Z가-힣0-9.,!?:;\'\" ]')
        self.tokenizer = tokenizer

    def __call__(self, sen) :
        assert isinstance(sen, str)
        tok_list = self.tokenizer.morphs(sen)
        sen = ' '.join(tok_list)
        sen = self.filter.sub('', sen)
        return sen
 