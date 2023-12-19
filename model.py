import torch
import torch.nn.functional as F
import torch.nn as nn

class HMModel(nn.Module):
    def __init__(self, article_shape):
        super(HMModel, self).__init__()
        
        self.article_emb = nn.Embedding(article_shape[0], embedding_dim=article_shape[1])
        self.article_likelihood = nn.Parameter(torch.zeros(article_shape[0]), requires_grad=True)

        self.top = nn.Sequential(nn.Conv1d(3, 16, kernel_size=1), nn.LeakyReLU(),
                                 nn.Conv1d(16, 8, kernel_size=1), nn.LeakyReLU(),
                                 nn.Conv1d(8, 1, kernel_size=1))
        
    def forward(self, inputs):
        article_hist, week_hist = inputs[0], inputs[1]
        x = self.article_emb(article_hist)
        x = F.normalize(x, dim=2)
        
        x = x@F.normalize(self.article_emb.weight).T
        
        x, indices = x.max(axis=1)
        x = x.clamp(1e-3, 0.999)
        x = -torch.log(1/x - 1)
        
        max_week = week_hist.unsqueeze(2).repeat(1, 1, x.shape[-1]).gather(1, indices.unsqueeze(1).repeat(1, week_hist.shape[1], 1))
        max_week = max_week.mean(axis=1).unsqueeze(1)
        
        x = torch.cat([x.unsqueeze(1), max_week,
                       self.article_likelihood[None, None, :].repeat(x.shape[0], 1, 1)], axis=1)
        
        x = self.top(x).squeeze(1)
        return x
    
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv1d(8, 16, kernel_size=8)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=8)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=8)
        
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv4 = nn.Conv1d(64, 64, kernel_size=8)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=8)
        
        self.bn2 = nn.BatchNorm1d(128)

        self.lstm1 = nn.LSTM(12, 100)
        self.lstm2 = nn.LSTM(100, 128)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
            
    def exec_conv_block(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = F.max_pool1d(x, 2)
        x = self.bn1(x)
        
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = F.max_pool1d(x, 2)
        x = self.bn2(x)
            
        return x
    
    def forward(self, x):
        x = self.exec_conv_block(x)

        x, state = self.lstm1(x)
        x, _ = self.lstm2(x, state)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x