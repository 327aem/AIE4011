import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from multiprocessing import freeze_support
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataset import *
from model import *
from utils import *


DATA_PATH = './data'
df = pd.read_csv(f"{DATA_PATH}/transactions_train.csv", dtype={"article_id": str})
# print(len(df))
# df = df.head(1000000)
df["t_dat"] = pd.to_datetime(df["t_dat"])

active_articles = df.groupby("article_id")["t_dat"].max().reset_index()
active_articles = active_articles[active_articles["t_dat"] >= "2019-08-24"].reset_index()

df = df[df["article_id"].isin(active_articles["article_id"])].reset_index(drop=True)

df["week"] = (df["t_dat"].max() - df["t_dat"]).dt.days // 7

article_ids = np.concatenate([["placeholder"], np.unique(df["article_id"].values)])

le_article = LabelEncoder()
le_article.fit(article_ids)
df["article_id"] = le_article.transform(df["article_id"])



val_weeks = [0]
train_weeks = [1, 2, 3]
WEEK_HIST_MAX = 5

# memory save
df = df[df.week < (max(train_weeks) + WEEK_HIST_MAX) * 2].reset_index(drop=True)

val_df = pd.concat([create_dataset(df, w) for w in val_weeks]).reset_index(drop=True)
train_df = pd.concat([create_dataset(df, w) for w in train_weeks]).reset_index(drop=True)

cons_users = list(set(train_df.customer_id).union(set(val_df.customer_id)))




def train(model, train_loader, val_loader, epochs):
    np.random.seed(SEED)
    
    optimizer = get_optimizer(model)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e1, 
                                              max_lr=5e-4, epochs=epochs, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()
    best_score = 0
    
    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
                
        loss_list = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                loss = criterion(logits, target) + dice_loss(logits, target)
            
            
            #loss.backward()
            scaler.scale(loss).backward()
            #optimizer.step()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            loss_list.append(loss.detach().cpu().item())
            
            avg_loss = np.round(100*np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e+1} Loss: {avg_loss}")
            
        val_map = validate(model, val_loader)

        log_text = f"Epoch {e+1}\nTrain Loss: {avg_loss}\nValidation MAP: {val_map}\n"
        print(log_text)
        if val_map > best_score:
            torch.save(model.state_dict(), 'best-model-parameters.pt')
            best_score = val_map
    #model.load_state_dict(torch.load('best-model-parameters.pt')) 
    
    return model

if __name__ == "__main__":
    freeze_support()
    SEQ_LEN = 16
    BS = 256
    NW = 8
    MODEL_NAME = "exp001"
    SEED = 0
    article_emb_size = 256

    # model = HMModel((len(le_article.classes_), article_emb_size))
    model = LSTM()
    model = model.cuda()

    val_dataset = HMDataset(val_df, SEQ_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False, num_workers=NW,
                            pin_memory=False, drop_last=False)

    train_dataset = HMDataset(train_df, SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=NW,
                            pin_memory=False, drop_last=True)

    model = train(model, train_loader, val_loader, epochs=3)