import os
import sys
import random
import argparse
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
from konlpy.tag import Mecab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from model import *
from tokenizer import *
from preprocessor import *
from scheduler import *

def progressLearning(value, endvalue, loss, acc, bar_length=50):
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent: [{0}] {1}/{2} \t Loss : {3:.3f}, Acc : {4:.3f}".format(arrow + spaces, 
        value+1, 
        endvalue, 
        loss, 
        acc)
    )
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args) :
    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- Text Data
    print('Load Data \n')
    train_data_path = os.path.join(args.data_dir, 'train_nsmc.csv')
    train_data = pd.read_csv(train_data_path).dropna()
    text_data = list(train_data['document'])

    # -- Preprocessor
    print('Load Preprocessor')
    mecab = Mecab()
    preprocessor = SenPreprocessor(mecab)
    text_preprocessed = [preprocessor(text) for text in tqdm(text_data)]
    print()

    # -- Tokenize & Encoder
    print('Load Tokenizer \n')
    text_path = os.path.join(args.data_dir, 'ratings_data.txt')
    if os.path.exists(text_path) == False :
        write_data(text_preprocessed, text_path)
        model_path = os.path.join(args.tokenizer_dir, 'ratings_tokenizer')
        train_spm(text_path, model_path, args.token_size)
    kor_tokenizer = get_spm(os.path.join(args.tokenizer_dir, 'ratings_tokenizer.model'))
    vocab_size = len(kor_tokenizer)

    print('Encode Data')
    idx_data = []
    for sen in tqdm(text_preprocessed) :
        idx_list = kor_tokenizer.encode_as_ids(sen)
        idx_data.append(idx_list)
    print('')

    # -- Dataset
    dset = ElmoDataset(idx_data, args.max_size)
    data_len = [len(data) for data in dset]
    data_collator = ElmoCollator(data_len, args.batch_size, args.backward_flag)
    
    # -- DataLoader
    data_loader = DataLoader(dset,
        num_workers=multiprocessing.cpu_count()//2,
        batch_sampler=data_collator.sample(),
        collate_fn=data_collator
    )
    
    # -- Model
    model = ElmoModel(layer_size=args.layer_size,
        vocab_size = vocab_size,
        embedding_size = args.embedding_size,
        hidden_size = args.hidden_size,
        cuda_flag = use_cuda,
    ).to(device)

    init_lr = 1e-4
    # -- Optimizer
    optimizer = optim.Adam(model.parameters(), 
        lr=init_lr, 
        betas=(0.9,0.98), 
        eps=1e-9, 
        weight_decay=1e-4
    )

    # -- Scheduler
    schedule_fn = Scheduler(args.embedding_size, init_lr, args.warmup_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: schedule_fn(epoch))

    # -- Logging
    writer = SummaryWriter(args.log_dir)

    # -- Loss
    criterion = nn.CrossEntropyLoss().to(device)

    # -- Training
    print('Training')
    log_count = 0
    model_name = 'lstm_backward.pt' if args.backward_flag else 'lstm_forward.pt'
    for epoch in range(args.epochs) :
        idx = 0
        mean_loss = 0.0
        mean_acc = 0.0
        model.train()
        print('Epoch : %d/%d' %(epoch, args.epochs))
        for data in data_loader :
            optimizer.zero_grad()
            writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], idx)

            in_data = data['in'].long().to(device)
            label_data = data['out'].long().to(device)
            label_data = torch.reshape(label_data, (-1,))

            out_data = model(in_data)
            out_data = torch.reshape(out_data, (-1,vocab_size))

            loss = criterion(out_data, label_data)
            acc = (torch.argmax(out_data,-1) == label_data).float().mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            mean_loss += loss
            mean_acc += acc

            progressLearning(idx, len(data_loader), loss.item(), acc.item())
            if (idx + 1) % 20 == 0 :
                writer.add_scalar('train/loss', loss.item(), log_count)
                writer.add_scalar('train/acc', acc.item(), log_count)
                log_count += 1
            idx += 1

        mean_loss /= len(data_loader)
        mean_acc /= len(data_loader)

        torch.save({'epoch' : (epoch) ,  
            'model_state_dict' : model.state_dict() , 
            'loss' : mean_loss.item(), 
            'acc' : mean_acc.item()},
        os.path.join(args.model_dir, model_name))
        print('\nMean Loss : %.3f , Mean Acc : %.3f\n' %(mean_loss, mean_acc))


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    # training environment
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--layer_size', type=int, default=3, help='layer size of lstm (default: 3)')
    parser.add_argument('--token_size', type=int, default=25000, help='number of bpe merge (default: 25000)')
    parser.add_argument('--max_size', type=int, default=64, help='max length of sequence (default: 64)')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding size of token (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=1024, help='hidden size of lstm (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--backward_flag', type=bool, default=False, help='flag of backward direction (default : False / Forward)')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warmup steps for training (default: 4000)')   

    # container environment
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--tokenizer_dir', type=str, default='./Tokenizer')
    parser.add_argument('--log_dir', type=str, default='./Log/pre_training')
    parser.add_argument('--model_dir', type=str, default='./Model/pre_training')
     
    args = parser.parse_args()
    train(args)
