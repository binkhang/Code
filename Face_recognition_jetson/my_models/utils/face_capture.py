import cv2
from my_models.mtcnn.mtcnn_model import MTCNN
# from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os

def capture_images(num_images, img_path):
    device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    leap = 1
    mtcnn = MTCNN(margin = 20, keep_all=False, select_largest = True, post_process=True, device = device) #post_process = true to normalize images captured
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    #===========================================================================================Review below point
    # cap.set(cv2.CAP_PROP_FPS, 4)

    # Check number of user's image in folder
    folder_count = len([name for name in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, name))])
    # Create new path for new user 
    usr_path = os.path.join(img_path,'user_' +str(folder_count))
    img_count = 0
    count = num_images
    while cap.isOpened() and count:
        isSuccess, frame = cap.read()
        if mtcnn(frame) is not None and leap%2:
            path = os.path.join(usr_path, '{}.jpg'.format('user_' + str(folder_count) +'_'+ str(img_count)))
            face_img = mtcnn(frame, save_path = path)
            img_count+=1
            count-=1
        leap+=1
        cv2.imshow('Face Capturing', frame)
        if cv2.waitKey(1)&0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    
