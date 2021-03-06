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
      1. Forward - Autoregressive
      2. Backward - Autoregressive
  3. Finetuning
      1. Text Classification

## Model Specification
  1. Layer size : 3
  2. Vocab size : 25000
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
      4. Weight Decay : 1e-2

## Finetuning
  1. Epoch size : 15
  2. Batch size : 128
  3. Optimizer : Adam
      1. Learning Rate : 1e-4
      2. Betas : (0.9, 0.98)
      3. Weight Decay : 1e-2
  4. Scheduler : Exponential Scheduler 
      1. Per Epoch
      2. Gamma = 0.8

