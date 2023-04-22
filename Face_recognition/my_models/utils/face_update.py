import glob
import torch 
from torchvision import transforms
# from my_models.facenet.facenet_model import InceptionResnetV1
# from my_models.mtcnn.mtcnn_model import fixed_image_standardization
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np
import openpyxl

def create_embeddings(IMG_PATH, DATA_PATH):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    def trans(img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        return transform(img)

    model = InceptionResnetV1(
        classify=False,
        pretrained="vggface2"
    ).to(device)

    model.eval()

    embeddings = []
    names = []

    for usr in os.listdir(IMG_PATH):
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
        embedding = torch.cat(embeds).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(usr)

    embeddings = torch.cat(embeddings)
    names = np.array(names)

    if device == 'cpu':
        torch.save(embeddings, os.path.join(DATA_PATH, "faceslistCPU.pth"))
    else:
        torch.save(embeddings, os.path.join(DATA_PATH, "faceslist.pth"))
    np.save(os.path.join(DATA_PATH, "usernames"), names)
    print(f'Update Completed! There are {names.shape[0]} people in FaceLists')


    embeddings = torch.load(os.path.join(DATA_PATH, "faceslist.pth"))

    workbook = openpyxl.Workbook()
    worksheet = workbook.active

    for i, embedding in enumerate(embeddings):
        for j, value in enumerate(embedding):
            cell = worksheet.cell(row=i+1, column=j+1)
            cell.value = value.item()

    workbook.save(os.path.join(DATA_PATH, "embeddings.xlsx"))
