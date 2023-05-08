import torch
import random
import os
import pandas as pd
import openpyxl
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_PATH = './Face_recognition/encoded_data'
workbook = openpyxl.Workbook()
worksheet = workbook.active   
# load embeddings
if device == 'cpu':
    tensors_list = torch.load(DATA_PATH+'/faceslistCPU.pth')
else:
    tensors_list = torch.load(DATA_PATH+'/faceslist.pth')

# return index which has sum of distance less


def Min_SumOfDistance(id1, id2, tensors):
    total_dist_id1 = 0.0
    total_dist_id2 = 0.0
    for k in range(len(tensors)):
        if k != id2:
            dists = (tensors[k] - tensors[id1]).norm().item()
            total_dist_id1 += dists
    for l in range(len(tensors)):
        if l != id1:
            dists = (tensors[l] - tensors[id2]).norm().item()
            total_dist_id2 += dists
    if (total_dist_id1 > total_dist_id2):
        return id2
    else:
        return id1


def Minimize_data(numOfData, tensors_list):
    while len(tensors_list) > numOfData:
        distance = []
        for e1 in range(len(tensors_list)):
            for e2 in range(e1+1, len(tensors_list)):
                dists = (tensors_list[e1] - tensors_list[e2]).norm().item()
                distance.append((e1, e2, dists))

        # Sắp xếp các cặp theo khoảng cách tăng dần
        sorted_distance = sorted(distance, key=lambda x: x[2])
        id1, id2, _ = sorted_distance[0]
        del_idx = Min_SumOfDistance(id1, id2, tensors_list)
        tensors_list = tuple(
            tensors_list[:del_idx]) + tuple(tensors_list[del_idx+1:])
    return tensors_list
names = []
minimize_list = Minimize_data(8,tensors_list)
for i, embedding in enumerate(minimize_list):
    names.append(0)
    for j, value in enumerate(embedding):
        cell = worksheet.cell(row=i+1, column=j+1)
        cell.value = value.item()
workbook.save(os.path.join(DATA_PATH, "minimize_embeddings.xlsx"))

names = np.array(names)
np.save(os.path.join(DATA_PATH, "usernames"), names)
print(f'Update Completed! There are {names.shape[0]} people in FaceLists')


if device == 'cpu':
    torch.save(minimize_list, os.path.join(DATA_PATH, "minimize_embeddings.pth"))
else:
    torch.save(minimize_list, os.path.join(DATA_PATH, "minimize_embeddings.pth"))

