# NSMC
## Modules
```
├── Log
│   ├── finetune
│   └── pretrain
├── Model
│   ├── finetune
│   └── pretrain
├── README.md
├── Token
├── dataset.py
├── model.py
├── pretrain.py
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
  2. Embedding size : 256
  3. Hidden size : 1024
  4. Sequence size : 30

## Train Specification
  1. Epoch size : 30
  2. Batch size : 128
  3. Learning rate : 1e-4
  4. Optimizer : Adam
  5. Scheduler : Exponential
