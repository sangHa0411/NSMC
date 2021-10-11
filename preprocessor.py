import re

class SenPreprocessor :
    def __init__(self) :
        self.filter = re.compile('[^a-zA-Z가-힣0-9.,!?:;\'\" ]')

    def __call__(self, sen) :
        assert isinstance(sen, str)
        sen = self.filter.sub('', sen)
        return sen
 
