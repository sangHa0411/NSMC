import os
import sys
import random
import argparse
import multiprocessing
import numpy as np
from tqdm import tqdm
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from model import *
from tokenizer import *

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
    print('Load Data')
    text_data = load_data(args.data_dir)

    # -- Tokenize & Encoder
    kor_text_path = os.path.join(args.token_dir, 'dialogue.txt')
    if os.path.exists(kor_text_path) == False :
        write_data(text_data, kor_text_path, preprocess_kor)
        train_spm(args.token_dir,  'dialogue.txt', 'kor_tokenizer' , args.token_size)
    kor_tokenizer = get_spm(args.token_dir, 'kor_tokenizer.model')
    vocab_size = len(kor_tokenizer)

    print('Encode Data')
    idx_data = []
    for sen in tqdm(text_data) :
        sen = preprocess_kor(sen)
        idx_list = kor_tokenizer.encode_as_ids(sen)
        idx_data.append(idx_list)

    # -- Dataset
    dset = ElmoDataset(idx_data, args.max_size)
    data_len = [len(data) for data in dset]
    data_collator = ElmoCollator(data_len, args.batch_size)
    
    # -- DataLoader
    data_loader = DataLoader(dset,
        num_workers=multiprocessing.cpu_count()//2,
        batch_sampler=data_collator.sample(),
        collate_fn=data_collator
    )
    
    # -- Model
    model = ElmoModel(layer_size=args.layer_size,
        v_size = vocab_size,
        em_size = args.embedding_size,
        h_size = args.hidden_size,
        cuda_flag = use_cuda,
        backward_flag=args.backward_flag
    ).to(device)

    # -- Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -- Logging
    writer = SummaryWriter(args.log_dir)

    # -- Loss
    criterion = nn.CrossEntropyLoss().to(device)

    # -- Training
    log_count = 0
    model_name = 'lstm_backward.pt' if args.backward_flag else 'lstm_forward.pt'
    for epoch in range(args.epochs) :
        idx = 0
        mean_loss = 0.0
        mean_acc = 0.0
        model.train()
        print('Epoch : %d/%d \t Learning Rate : %e' %(epoch, args.epochs, optimizer.param_groups[0]["lr"]))
        for data in data_loader :
            if args.backward_flag == True :
                in_data = data['out'].long().to(device)
                label_data = data['in'].long().to(device)
                label_data = torch.flip(label_data, [1])
                label_data = torch.reshape(label_data, (-1,))
            else :
                in_data = data['in'].long().to(device)
                label_data = data['out'].long().to(device)
                label_data = torch.reshape(label_data, (-1,))

            out_data = model(in_data)
            out_data = torch.reshape(out_data, (-1,vocab_size))

            loss = criterion(out_data, label_data)
            acc = (torch.argmax(out_data,-1) == label_data).float().mean()

            loss.backward()
            optimizer.step()

            mean_loss += loss
            mean_acc += acc

            progressLearning(idx, len(data_loader), loss.item(), acc.item())
            if (idx + 1) % 10 == 0 :
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

    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--layer_size', type=int, default=3, help='layer size of lstm (default: 3)')
    parser.add_argument('--token_size', type=int, default=32000, help='number of bpe merge (default: 32000)')
    parser.add_argument('--max_size', type=int, default=32, help='max length of sequence (default: 32)')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding size of token (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=1024, help='hidden size of lstm (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--backward_flag', type=bool, default=False, help='flag of backward direction (default : False / Forward)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')    

    parser.add_argument('--data_dir', type=str, default='../Word2Vec/Data', help = 'data path')
    parser.add_argument('--token_dir', type=str, default='./Token' , help='token data dir path')
    parser.add_argument('--log_dir', type=str, default='./Log/pretrain' , help='loggind data dir path')
    parser.add_argument('--model_dir', type=str, default='./Model/pretrain' , help='best model dir path')

    args = parser.parse_args()

    train(args)
