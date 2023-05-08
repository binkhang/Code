import os
from add_new_user import Reload_all_users
minimize_embed_path = './Face_recognition/encoded_data/minimized_embeddings'



def Delete_user():
    user_id = input("Enter the name of the file you want to delete: ")
    file_path = os.path.join(minimize_embed_path, f'user_{user_id}_min.pth')
    print(file_path)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{user_id} has been deleted.")
    else:
        print(f"user_{user_id} does not exist in {minimize_embed_path}.")
    
    Reload_all_users()

    

Delete_user()