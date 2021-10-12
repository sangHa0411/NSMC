# NSMC
## Modules
```
|-- Data
|-- Log
|-- Model
|   `-- lstm_forward.pt
|-- README.md
|-- Tokenizer
|   |-- ratings_tokenizer.model
|   `-- ratings_tokenizer.vocab
|-- dataset.py
|-- model.py
|-- preprocessor.py
|-- pretrain.py
|-- scheduler.py
|-- tokenizer.py
`-- train.py
```
  
## Model 
  1. ELMo
  2. Pretraining
      1. forward
      2. backward 
  3. Finetuning
      1. Text Classification

## Model Specification
  1. Layer size : 3
  2. Vocab size : 32000
  3. Embedding size : 256
  4. Hidden size : 1024
  5. Sequence size : 64

## Pretraining
  1. Epoch size : 30
  2. Batch size : 256
  4. Optimizer : Adam
      1. Betas = (0.9, 0.98)
      2. eps = 1e-9
  6. Warmup Steps : 4000

