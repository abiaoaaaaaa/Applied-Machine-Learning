import os
from pathlib import Path
import shutil

# Create folder
# path: folder path
def create_folder(path):
    if not os.path.exists(path):
      os.makedirs(path)
      print(f"Folder '{path}' created.")
    else:
      for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
      print(f"Folder '{path}' already exists.")

    

