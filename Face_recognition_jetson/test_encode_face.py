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



def gen_embedding(model, face, threshold = 3):
    #local: [n,512] voi n la so nguoi trong faceslist
    embeds = []
    # print(trans(face).unsqueeze(0).shape)
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds) #[1,512]
    print(detect_embeds)
    # print(detect_embeds.shape)
                    #[1,512,1]                                      [1,512,n]

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
    cv2.imshow('face after crop', face)
    face = Image.fromarray(face)
    return face

if __name__ == "__main__":
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = InceptionResnetV1(
        classify=False,
        pretrained="vggface2"
    ).to(device)
    model.eval()

    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    
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
        gen_embedding(model, face)

    # Display the image with the face detection and recognition results
    cv2.imshow('Face Recognition', frame)
    cv2.waitKey(0)

    # Release the video stream and close the window
    cap.release()
    cv2.destroyAllWindows()
    