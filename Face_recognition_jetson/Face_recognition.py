import cv2
from my_models.mtcnn.mtcnn_model import MTCNN, fixed_image_standardization
from my_models.facenet.facenet_model import InceptionResnetV1
# from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time

frame_size = (640,480)
IMG_PATH = './Face_recognition/img_users'
DATA_PATH = './Face_recognition/encoded_data'

def trans(img):
    transform = transforms.ToTensor()
    return transform(img)

def load_faceslist(): 
    if device == 'cpu':
        embeds = torch.load(DATA_PATH+'/embeddings.pth')
    else:
        embeds = torch.load(DATA_PATH+'/embeddings.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    # print (names)
    return embeds, names

def inference(model, face, local_embeds, threshold = 0.95): 
    #local: [n,512] voi n la so nguoi trong faceslist
    embeds = []
    # print(trans(face).unsqueeze(0).shape)
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds) #[1,512]
    print(detect_embeds.shape)
                    #[1,512,1]                                      [1,512,n]
    norm_score = []
    for i in range (len(local_embeds)):
        dist = (detect_embeds - local_embeds[i]).norm().item()
        norm_score.append((i,dist))
    norm_score = torch.tensor(norm_score)
    sorted_norm_score = sorted(norm_score, key=lambda x: x[1])
    print(norm_score) 
    #knn with k == 1
    embed_idx, min_dist = sorted_norm_score[0]
    print(embed_idx)
    print(min_dist)

    if min_dist > threshold:
        return -1, -1
    else:
        return int(embed_idx.item()), min_dist.double()

def extract_face(box, img, margin=20):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ] #tạo margin bao quanh box cũ
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face

if __name__ == "__main__":
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    resnet = InceptionResnetV1(
        classify=False,
        pretrained="vggface2"
    ).to(device)
    resnet.eval()

    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    embeddings, names = load_faceslist()

    
    ret, frame = cap.read()
    if not ret:
        print('Failed to capture image ')
        cap.release()
        cv2.destroyAllWindows()
        exit()
    
    # Detect faces in the image
    boxes, _ = mtcnn.detect(frame)
    while boxes is None:
        ret, frame = cap.read()
        boxes, _ = mtcnn.detect(frame)
        print('No faces detected')

    # Perform face recognition on each detected face
    for box in boxes:
        bbox = list(map(int, box.tolist()))
        face = extract_face(bbox, frame)
        idx, score = inference(resnet, face, embeddings)
        print(idx)
        if idx != -1:
            # print(idx)
            frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
            # score = torch.Tensor.cpu(score[0]).detach().numpy()*power
            frame = cv2.putText(frame, str(names[idx]), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
        else:
            frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
            frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)

    # Display the image with the face detection and recognition results
    cv2.imshow('Face Recognition', frame)
    cv2.waitKey(0)

    # Release the video stream and close the window
    cap.release()
    cv2.destroyAllWindows()
    