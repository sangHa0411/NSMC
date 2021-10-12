import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter
from konlpy.tag import Mecab
import sys
import re
import os
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import random
from importlib import import_module
from dataset import *
from model import *
from preprocessor import *

def progressLearning(value, endvalue, loss , acc , bar_length=50):
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent: [{0}] {1}/{2} \t Loss : {3:.3f} , Acc : {4:.3f}".format(arrow + spaces, value+1 , endvalue , loss , acc))
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def acc_fn(y_output, y_label) :
    y_arg = torch.where(y_output>=0.5, 1.0, 0.0)
    y_acc = (y_arg == y_label).float()
    y_acc = torch.mean(y_acc)
    return y_acc
    
def evaluate(model, test_loader, critation, device) :
    with torch.no_grad() :
        model.eval()
        loss_eval = 0.0
        acc_eval = 0.0
    
        for idx_data, label_data in test_loader :
            idx_data = idx_data.long().to(device)
            label_data = label_data.to(device)

            out_data = model(idx_data)
            
            loss_eval += critation(out_data , label_data)
            acc_eval += acc_fn(out_data , label_data)

        model.train()
        loss_eval /= len(test_loader)
        acc_eval /= len(test_loader)
        
    return loss_eval , acc_eval  

def train(args) :

    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- Raw Train Data
    train_data = pd.read_table(args.train_data_dir).dropna()

    train_processor = Preprocessor(train_data['document'], mecab.morphs)
    token_data = train_processor.get_data()
    train_index = train_processor.encode()
    train_label = list(train_data['label'])
    
    # -- Raw Test Data
    test_data = pd.read_table(args.test_data_dir).dropna()
    test_data = preprocess_data(test_data, 10)
    
    test_processor = Preprocessor(test_data['document'], mecab.morphs)
    test_processor.set_data(token_data)
    test_index = test_processor.encode()
    test_label = list(test_data['label'])

    # -- Dataset
    train_dset = BaseDataset(train_index, train_label, args.sen_size)
    train_len = train_dset.get_size()
    train_collator = Collator(train_len, args.batch_size)

    test_dset = BaseDataset(test_index, test_label, args.sen_size)
    test_len = test_dset.get_size()
    test_collator = Collator(test_len, args.batch_size)

    # -- Dataloader
    train_loader = DataLoader(train_dset,
        num_workers=multiprocessing.cpu_count()//2,
        batch_sampler=train_collator.batch_sampler(),
        collate_fn = train_collator
    )

    test_loader = DataLoader(test_dset,
        num_workers=multiprocessing.cpu_count()//2,
        batch_sampler=test_collator.batch_sampler(),
        collate_fn=test_collator
    )

    v_size = len(token_data)
    # -- Classification Model
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        layer_size = args.layer_size, 
        h_size = args.embedding_size, 
        v_size = v_size,
        use_cuda = use_cuda
    ).to(device)

    # -- Optimizer
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  
    optimizer = opt_module(
        model.parameters(),
        lr=args.lr
    )

    # -- Scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # -- Logging
    writer = SummaryWriter(args.log_dir)

    # -- loss
    loss_fn = nn.BCELoss().to(device)
    
    print('Training Starts')
    min_loss = np.inf
    stop_count = 0
    log_count = 0
    # -- Training
    for epoch in range(args.epochs) :
        print('Epoch : %d' %epoch)
        idx = 0
        for idx_data, label_data in train_loader :
            
            optimizer.zero_grad()

            idx_data = idx_data.long().to(device)
            label_data = label_data.to(device)
            out_data = model(idx_data)

            loss = loss_fn(out_data , label_data)
            acc = acc_fn(out_data , label_data)

            loss.backward()
            optimizer.step()
        
            progressLearning(idx, len(train_loader), loss.item(), acc.item())

            if (idx + 1) % 10 == 0 :
                writer.add_scalar('train/loss', loss.item(), log_count)
                writer.add_scalar('train/acc', acc.item(), log_count)
                log_count += 1
            idx += 1

        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device) 
       
        if test_loss < min_loss :
            min_loss = test_loss
            torch.save({'epoch' : (epoch) ,  
                        'model_state_dict' : model.state_dict() , 
                        'loss' : test_loss.item() , 
                        'acc' : test_acc.item()} , 
                        os.path.join(args.model_dir, 'nsmc_model.pt'))        
            stop_count = 0 
        else :
            stop_count += 1
            if stop_count >= 5 :      
                print('\nTraining Early Stopped')
                break
        scheduler.step()
        print('\nVal Loss : %.3f Val Accuracy : %.3f \n' %(test_loss, test_acc))
    print('Training finished')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 20)')
    parser.add_argument('--sen_size', type=int, default=30, help='max sentence size (default: 30)')
    parser.add_argument('--layer_size', type=int, default=3, help='layer size of model (default: 3)')
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding size of token (default: 128)')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--val_batch_size', type=int, default=256, help='input batch size for validing (default: 256)')
    parser.add_argument('--model', type=str, default='StackedLSTM', help='model type (default: StackedLSTMl)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')    

    # Container environment
    parser.add_argument('--train_data_dir', type=str, default='./ratings_train.txt')
    parser.add_argument('--test_data_dir', type=str, default='./ratings_test.txt')
    parser.add_argument('--tokenizer_dir', type=str, default='./Tokenizer')
    parser.add_argument('--model_dir', type=str, default='./Model')
    parser.add_argument('--log_dir' , type=str , default='./Log')

    args = parser.parse_args()

    train(args)

