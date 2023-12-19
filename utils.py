import torch
import numpy as np
import sys
from tqdm import tqdm

def calc_map(topk_preds, target_array, k=12):
    metric = []
    tp, fp = 0, 0
    
    for pred in topk_preds:
        if target_array[pred]:
            tp += 1
            metric.append(tp/(tp + fp))
        else:
            fp += 1
            
    return np.sum(metric) / min(k, target_array.sum())

def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader, k=12):
    model.eval()
    
    tbar = tqdm(val_loader, file=sys.stdout)
    
    maps = []
    
    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            logits = model(inputs)

            _, indices = torch.topk(logits, k, dim=1)

            indices = indices.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            for i in range(indices.shape[0]):
                maps.append(calc_map(indices[i], target[i]))
        
    
    return np.mean(maps)

def dice_loss(y_pred, y_true):
    y_pred = y_pred.sigmoid()
    intersect = (y_true*y_pred).sum(axis=1)
    
    return 1 - (intersect/(intersect + y_true.sum(axis=1) + y_pred.sum(axis=1))).mean()

def get_optimizer(net):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4, betas=(0.9, 0.999),
                                 eps=1e-08)
    return optimizer