import glob
import torch 
from torchvision import transforms
from my_models.facenet.facenet_model import InceptionResnetV1
from my_models.mtcnn.mtcnn_model import fixed_image_standardization
import pandas as pd
# from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np
import openpyxl

def create_embeddings(IMG_PATH, DATA_PATH):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # def trans(img):
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         fixed_image_standardization
    #     ])
    #     return transform(img)
    def trans(img):
        transform = transforms.ToTensor()
        return transform(img)

    model = InceptionResnetV1(
        classify=False,
        pretrained="vggface2"
    ).to(device)

    model.eval()

    embeddings = []
    names = []

    for usr in os.listdir(IMG_PATH):
        # print(usr)
        embeds = []
        for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
            try:
                img = Image.open(file)
            except:
                continue
            with torch.no_grad():
                embeds.append(model(trans(img).to(device).unsqueeze(0)))
        if len(embeds) == 0:
            continue
        # embedding = torch.cat(embeds).mean(0, keepdim=True)
        embedding = torch.cat(embeds)
        embeddings.append(embedding)
        # print(embedding)
        names.append(usr)

    embeddings = torch.cat(embeddings)
    names = np.array(names)
    #caculate distance
    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    df = pd.DataFrame(dists)
    print(df)
    df.to_excel('Face_recognition/encoded_data/Temp_Dist.xlsx', index= True)

    if device == 'cpu':
        torch.save(embeddings, os.path.join(DATA_PATH, "faceslistCPU.pth"))
    else:
        torch.save(embeddings, os.path.join(DATA_PATH, "faceslist.pth"))
    np.save(os.path.join(DATA_PATH, "usernames"), names)
    print(f'Update Completed! There are {names.shape[0]} people in FaceLists')


