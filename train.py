import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from importlib import import_module
import sys
import re
import os
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import random

from dataset import *
from model import *
from tokenizer import *
from preprocessor import *
from scheduler import *

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
    
def train(args) :

    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- Raw Train Data
    print('Load Train Data\n')
    train_data = pd.read_csv(args.train_data_dir).dropna()
    train_text = list(train_data['document'])
    train_label = list(train_data['label'])
    
    # -- Raw Test Data
    print('Load Test Data\n')
    test_data = pd.read_csv(args.test_data_dir).dropna()
    test_text = list(test_data['document'])
    test_label = list(test_data['label'])
    
    # -- Preprocessor
    print('Preprocessing')
    sen_preprocessor = SenPreprocessor()
    train_text = [sen_preprocessor(sen) for sen in tqdm(train_text)]
    test_text = [sen_preprocessor(sen) for sen in tqdm(test_text)]
    print('\n')

    # -- Encoder
    print('Load Tokenizer\n')
    tokenizer = get_spm(os.path.join(args.tokenizer_dir, 'ratings_tokenizer.model'))
    vocab_size = len(tokenizer)
    
    print('Encoding')
    train_idx = []
    test_idx = []
    for sen in tqdm(train_text) :
        idx_list = tokenizer.encode_as_ids(sen)
        train_idx.append(idx_list)
    for sen in tqdm(test_text) :
        idx_list = tokenizer.encode_as_ids(sen)
        test_idx.append(idx_list)
    print('\n')

    # -- Dataset
    train_dset = NsmcDataset(train_idx, train_label, args.max_size)
    train_len = train_dset.get_size()
    train_collator = NsmcCollator(train_len, args.batch_size)

    test_dset = NsmcDataset(test_idx, test_label, args.max_size)
    test_len = test_dset.get_size()
    test_collator = NsmcCollator(test_len, args.batch_size)

    # -- Dataloader
    train_loader = DataLoader(train_dset,
        num_workers=multiprocessing.cpu_count()//2,
        batch_sampler=train_collator.sample(),
        collate_fn = train_collator
    )
    test_loader = DataLoader(test_dset,
        num_workers=multiprocessing.cpu_count()//2,
        batch_sampler=test_collator.sample(),
        collate_fn = test_collator
    )

    # -- Model
    print('Load Fowrad Model')
    forward_model = ElmoModel(layer_size=args.layer_size,
        vocab_size = vocab_size,
        embedding_size = args.embedding_size,
        hidden_size = args.hidden_size,
        cuda_flag = use_cuda,
    ).to(device)
    forward_checkpoint = torch.load(os.path.join(args.model_dir, 'pre_training', 'lstm_forward.pt'))
    forward_model.load_state_dict(forward_checkpoint['model_state_dict'])
    print('Load Backward Model')
    backward_model = ElmoModel(layer_size=args.layer_size,
        vocab_size = vocab_size,
        embedding_size = args.embedding_size,
        hidden_size = args.hidden_size,
        cuda_flag = use_cuda,
    ).to(device)
    backward_checkpoint = torch.load(os.path.join(args.model_dir, 'pre_training', 'lstm_backward.pt'))
    backward_model.load_state_dict(backward_checkpoint['model_state_dict'])

    # -- Classification Model
    print('Load Binary Classification Model')
    model = NsmcClassification(forward_model, backward_model, 1).to(device)

    # -- Optimizer
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    assert args.optimizer in ['SGD', 'Adam'] 
    if args.optimizer == 'SGD' :    
        optimizer = opt_module(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=1e-2
        )
    else :
        optimizer = opt_module(
            model.parameters(),
            lr=args.lr,
            betas=(0.9,0.98),
            eps=1e-9,
            weight_decay=1e-2
        )

    # -- Scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # -- Logging
    writer = SummaryWriter(args.log_dir)

    # -- loss
    critation = nn.BCELoss().to(device)
    
    print('Training Starts')
    min_loss = np.inf
    stop_count = 0
    log_count = 0
    # -- Training
    for epoch in range(args.epochs) :
        print('Epoch : %d\%d \t Learning Rate : %e' %(epoch, args.epochs, optimizer.param_groups[0]["lr"]))
        idx = 0
        for idx_data, label_data in train_loader :
            optimizer.zero_grad()
            writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], idx)

            idx_data = idx_data.long().to(device)
            label_data = label_data.float().to(device)
            out_data = model(idx_data)

            loss = critation(out_data , label_data)
            acc = acc_fn(out_data , label_data)

            loss.backward()
            optimizer.step()
        
            progressLearning(idx, len(train_loader), loss.item(), acc.item())
            if (idx + 1) % 100 == 0 :
                writer.add_scalar('train/loss', loss.item(), log_count)
                writer.add_scalar('train/acc', acc.item(), log_count)
                log_count += 1
            idx += 1

        with torch.no_grad() :
            model.eval()
            test_loss = 0.0
            test_acc = 0.0

            for idx_data, label_data in test_loader :
                idx_data = idx_data.long().to(device)
                label_data = label_data.float().to(device)
                out_data = model(idx_data)

                test_loss += critation(out_data , label_data)
                test_acc += acc_fn(out_data , label_data)

            model.train()
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)

        if test_loss < min_loss :
            min_loss = test_loss
            torch.save({'epoch' : (epoch) ,  
                        'model_state_dict' : model.state_dict() , 
                        'loss' : test_loss.item() , 
                        'acc' : test_acc.item()} , 
                        os.path.join(args.model_dir, 'fine_tuning', 'nsmc_model.pt'))        
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

    # Training environment
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--max_size', type=int, default=64, help='max sentence size (default: 64)')
    parser.add_argument('--layer_size', type=int, default=3, help='layer size of model (default: 3)')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding size of token (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=1024, help='lstm unit size of model (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--val_batch_size', type=int, default=256, help='input batch size for validing (default: 256)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate of training') 

    # Container environment
    parser.add_argument('--train_data_dir', type=str, default='./Data/train_nsmc.csv')
    parser.add_argument('--test_data_dir', type=str, default='./Data/test_nsmc.csv')
    parser.add_argument('--tokenizer_dir', type=str, default='./Tokenizer')
    parser.add_argument('--model_dir', type=str, default='./Model')
    parser.add_argument('--log_dir' , type=str , default='./Log/fine_tuning')

    args = parser.parse_args()
    train(args)

