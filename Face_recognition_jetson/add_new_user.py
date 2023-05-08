import cv2
from my_models.mtcnn.mtcnn_model import MTCNN 
import torch
from datetime import datetime
import os
import glob
from torchvision import transforms
from my_models.facenet.facenet_model import InceptionResnetV1
import pandas as pd
from PIL import Image
import numpy as np
import openpyxl

num_images = 40
numOfData = 8 #number of data after minimize / number of label
img_path = './Face_recognition/img_users'
embed_path = './Face_recognition/encoded_data/user_embeddings'
minimize_embed_path = './Face_recognition/encoded_data/minimized_embeddings'
data_path = './Face_recognition/encoded_data'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#define model FaceNet
model = InceptionResnetV1(
    classify=False,
    pretrained="vggface2"
).to(device)

def capture_images():
    leap = 1
    mtcnn = MTCNN(margin = 20, keep_all=False, select_largest = True, post_process=True, device = device) #post_process = true to normalize images captured
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    #===========================================================================================Review below point
    # cap.set(cv2.CAP_PROP_FPS, 4)
    
    # input the user's ID
    user_id = input("Enter your ID: ")
    while os.path.exists(os.path.join(img_path,f'user_{user_id}')) == True:
        print("ID already exists")
        user_id = input("Enter your ID: ")
    
    # Create new path for new user 
    usr_path = os.path.join(img_path,'user_' +user_id )
    img_count = 0
    count = num_images
    while cap.isOpened() and count:
        isSuccess, frame = cap.read()
        if mtcnn(frame) is not None and leap%2:
            path = os.path.join(usr_path, '{}.jpg'.format('user_' + user_id +'_'+ str(img_count)))
            face_img = mtcnn(frame, save_path = path)
            img_count+=1
            count-=1
        leap+=1
        cv2.imshow('Face Capturing', frame)
        if cv2.waitKey(1)&0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return user_id
def trans(img):
        transform = transforms.ToTensor()
        return transform(img)
# calculate 40 embeds for user which has ID returned by capture_images()
def create_embeddings(ID):
    model.eval()
    embeds = []
    for file in glob.glob(os.path.join(img_path, f'user_{ID}', '*.jpg')):
        try:
            img = Image.open(file)
        except:
            continue
        with torch.no_grad():
            embeds.append(model(trans(img).to(device).unsqueeze(0)))

    embedding = torch.cat(embeds)
    torch.save(embedding, os.path.join(embed_path, f'user_{ID}.pth'))
# minimize the number of embeds (40 - 8)
def Minimize_data(ID):
    tensors_list = torch.load(os.path.join(embed_path,f'user_{ID}.pth'))
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
        torch.save(tensors_list, os.path.join(minimize_embed_path, f'user_{ID}_min.pth'))

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

def Reload_all_users():
    embeds_list = []  
    names = []
    for filename in os.listdir(minimize_embed_path):
        if filename.endswith(".pth"):
            embeds = torch.load(os.path.join(minimize_embed_path, filename))
            if isinstance(embeds, tuple):
                # if the loaded object is a tuple of tensors, concatenate them element-wise
                embeds = tuple(torch.cat((t,), dim=0) for t in embeds)
            embeds_list.append(embeds)
            name = filename[:-8]
            for i in range(numOfData):
                names.append(name)
    # Concatenate all the embeds into a embed tensor
    concatenated_embeds  = [t for tup in embeds_list for t in tup]
    # Save the concatenated tensor to a file
    torch.save(concatenated_embeds, os.path.join(data_path, 'embeddings.pth'))
    np.save(os.path.join(data_path, "usernames"), names)
    print(names)
    print(f"There are {len(embeds_list)} in list")

def Add_user():
    user = capture_images()
    create_embeddings(user)
    Minimize_data(user)
    Reload_all_users()

