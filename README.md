# NSMC
## Modules
```
├── Data
│   ├── test_nsmc.csv
│   └── train_nsmc.csv
├── dataset.py
├── Log
│   ├── fine_tuning
│   └── pre_training
├── Model
│   ├── fine_tuning
│   └── pre_training
├── model.py
├── preprocessor.py
├── pretrain.py
├── README.md
├── scheduler.py
├── Tokenizer
│   ├── ratings_tokenizer.model
│   └── ratings_tokenizer.vocab
├── tokenizer.py
└── train.py
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
      3. Warmup Steps : 4000

## Finetuning
  1. Epoch size : 15
  2. Batch size : 128
  3. Optimizer : SGD
      1. Learning Rate : 3e-5
      2. Momentum : 0.9

