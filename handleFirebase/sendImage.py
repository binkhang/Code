import firebase_admin
from firebase_admin import db
from firebase_admin import credentials
from firebase_admin import storage
import os
import datetime

#Firebase Admin SDK
cred = credentials.Certificate("C:/Users/Admin/Desktop/ĐATN/Python_learn/SendImage/sendimage-71a2a-firebase-adminsdk-2zx6f-3584a5ec6e.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'sendimage-71a2a.appspot.com',
    'databaseURL': 'https://sendimage-71a2a-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

#Initial
bucket = storage.bucket()

# Thời gian hiện tại
now = datetime.datetime.now()

# Tạo tên folder là thời gian up ảnh
folder_name = now.strftime('%Y-%m-%d_%H-%M-%S')

# Tạo folder trên Realtime Database
folder_ref = db.reference(f'{folder_name}')

#upload image
filename = "C:/Users/Admin/Desktop/ĐATN/Python_learn/SendImage/avt1.jpg"
basename = os.path.basename(filename) 
name, ext = os.path.splitext(basename)
new_name = f"{name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}{ext}"

# Tạo blob mới với tên file mới
blob = bucket.blob(new_name)
blob.upload_from_filename(filename)

# Get download URL of image file
image_url = blob.generate_signed_url(expiration=300, method='GET')

# Store download URL in Realtime Database
folder_ref.set({
  'url': image_url
  #'url': image_url
})

