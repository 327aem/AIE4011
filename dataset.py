
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

val_weeks = [0]
train_weeks = [1, 2, 3]
WEEK_HIST_MAX = 5

DATA_PATH = './data'
df = pd.read_csv(f"{DATA_PATH}/transactions_train.csv", dtype={"article_id": str})

df["t_dat"] = pd.to_datetime(df["t_dat"])

active_articles = df.groupby("article_id")["t_dat"].max().reset_index()
active_articles = active_articles[active_articles["t_dat"] >= "2019-08-24"].reset_index()

df = df[df["article_id"].isin(active_articles["article_id"])].reset_index(drop=True)

df["week"] = (df["t_dat"].max() - df["t_dat"]).dt.days // 7

article_ids = np.concatenate([["placeholder"], np.unique(df["article_id"].values)])

le_article = LabelEncoder()
le_article.fit(article_ids)

def create_dataset(df, week):
    hist_df = df[(df["week"] > week) & (df["week"] <= week + WEEK_HIST_MAX)]
    hist_df = hist_df.groupby("customer_id").agg({"article_id": list, "week": list}).reset_index()
    hist_df.rename(columns={"week": 'week_history'}, inplace=True)
    
    target_df = df[df["week"] == week]
    target_df = target_df.groupby("customer_id").agg({"article_id": list}).reset_index()
    target_df.rename(columns={"article_id": "target"}, inplace=True)
    target_df["week"] = week
    
    return target_df.merge(hist_df, on="customer_id", how="left")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_data(filename):
    data = pd.read_csv(f"data/{filename}.csv")
    return data


def clean_customer_data(data):
    # Deal with NaN
    data['FN'] = data['FN'].fillna(0)
    data['FN'] = data['FN'].astype(int)

    data['Active'] = data['Active'].fillna(0)
    data['Active'] = data['Active'].astype(int)

    data['age'] = data['age'].fillna(-1)
    data['age'] = data['age'].astype(int)

    data['fashion_news_frequency'] = data['fashion_news_frequency'].fillna("None")

    data['club_member_status'] = data['club_member_status'].fillna(0)
    # One Hot Encode
    data['fashion_news_frequency'].mask(data['fashion_news_frequency'] == 'NONE', 'None', inplace=True)
    data = pd.concat([data, pd.get_dummies(data["fashion_news_frequency"], prefix='news_frequency')], axis=1)
    data['club_member_status'].mask(data['club_member_status'] == 'LEFT CLUB', 'left', inplace=True)
    data['club_member_status'].mask(data['club_member_status'] == 'PRE-CREATE', 'pre_create', inplace=True)
    data = pd.concat([data, pd.get_dummies(data["club_member_status"], prefix='status')], axis=1)
    # Drop some columns
    data.drop(["postal_code", "fashion_news_frequency", "club_member_status"], axis=1, inplace=True)
    data = change_data_types_to_save_memory(data)
    return data


def clean_article_data(data):
    data.drop(["detail_desc"], axis=1, inplace=True)

    data['article_id'] = data['article_id'].astype('string')

    data = change_data_types_to_save_memory(data)
    return data


def clean_transaction_data(data):
    data = data.rename(columns={'t_dat': 'date'})
    data["date"] = pd.to_datetime(data["date"])

    data['customer_id'] = data['customer_id'].astype('string')
    data['article_id'] = data['article_id'].astype('string')

    data['sales_channel_id'] = data['sales_channel_id'].astype('uint8')

    data = change_data_types_to_save_memory(data)
    return data


def change_data_types_to_save_memory(data):
    for col in data.select_dtypes(include=['int64']).columns:
        data[col] = data[col].astype('int32')
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category')
    return data


def add_num_sold_to_articles(transactions_df, articles_df):
    num_sold = transactions_df.groupby(['article_id']).size().reset_index(name='num_sold')
    articles_df = articles_df.merge(num_sold, how='left', on='article_id')
    articles_df['num_sold'] = articles_df['num_sold'].fillna(0)
    articles_df['num_sold'] = articles_df['num_sold'].astype('int32')
    return articles_df


def remove_old_articles(transactions_df, articles_df):
    # Get a list of only article ids that have sold recently
    last_transaction_cutoff = "2019-09-01"
    active_articles = transactions_df.groupby("article_id")["date"].max().reset_index()
    active_articles = active_articles[active_articles["date"] >= last_transaction_cutoff].reset_index()
    # Prune inactive articles from transactions and articles
    transactions_df = transactions_df[transactions_df["article_id"].isin(active_articles["article_id"])].reset_index(
        drop=True)
    articles_df = articles_df[articles_df["article_id"].isin(active_articles["article_id"])].reset_index(drop=True)
    return transactions_df, articles_df



def generate_correlation(df):
    plt.figure(figsize=[7, 5])
    sns.heatmap(df.corr())
    plt.show()
    

class HMDataset(Dataset):
    def __init__(self, df, seq_len, is_test=False):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.is_test = is_test
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        if self.is_test:
            target = torch.zeros(2).float()
        else:
            target = torch.zeros(len(article_ids)).float()
            for t in row.target:
                target[t] = 1.0
            
        article_hist = torch.zeros(self.seq_len).long()
        week_hist = torch.ones(self.seq_len).float()
        
        
        if isinstance(row.article_id, list):
            if len(row.article_id) >= self.seq_len:
                article_hist = torch.LongTensor(row.article_id[-self.seq_len:])
                week_hist = (torch.LongTensor(row.week_history[-self.seq_len:]) - row.week)/WEEK_HIST_MAX
            else:
                article_hist[-len(row.article_id):] = torch.LongTensor(row.article_id)
                week_hist[-len(row.article_id):] = (torch.LongTensor(row.week_history) - row.week)/WEEK_HIST_MAX
                
        return article_hist, week_hist, target
    