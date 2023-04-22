from my_models.utils.face_capture import capture_images
from my_models.utils.face_update import create_embeddings
# Example usage
num_images = 40
img_path = './Face_recognition/img_users'
data_path = './Face_recognition/encoded_data'
# capture_images(num_images, img_path)
create_embeddings(img_path,data_path)
