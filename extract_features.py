from Facenet import Facenet
from os import listdir
from os.path import isfile, isdir, join
from tqdm import tqdm
import numpy as np
import cv2
import pickle


# # BASE_LEFT = r"D:\WORK\Periocular-Recognition\images\ubipr\left\l2"
# BASE_RIGHT = r"D:\WORK\Periocular-Recognition\images\ubipr\right\r2"

# # lfiles = os.listdir(BASE_LEFT)
# rfiles = os.listdir(BASE_RIGHT)

# # print(len(lfiles))
# # print(len(rfiles))

# facenet = Facenet()

# # eye_images = [cv2.imread(os.path.join(BASE_LEFT, f)) for f in tqdm(lfiles)]
# # eye_features = facenet.get_embeddings(eye_images, verbose=1)
# # labels = [f.split("_")[0] for f in lfiles]

# # print(len(labels))
# # print(eye_features.shape)
    
# # fn = "data\\leye_features2.pkl"
# # with open(fn, "wb") as fl:
# #     pickle.dump({"features": eye_features, "labels": labels}, fl)

# # print(f"{fn} saved...")




# labels = []
# for f in rfiles:
#     c = f.split("_")[0][1:]
#     labels.append(f"C{int(c)-1}")

# eye_images = [cv2.imread(os.path.join(BASE_RIGHT, f)) for f in tqdm(rfiles)]
# eye_features = facenet.get_embeddings(eye_images, verbose=1)

# print(len(labels))
# print(eye_features.shape)
    
# fn = "data\\reye_features2.pkl"
# with open(fn, "wb") as fl:
#     pickle.dump({"features": eye_features, "labels": labels}, fl)

# print(f"{fn} saved...")

# BASE_DIR = r"D:\WORK\Periocular-Recognition\images\essex_data\data_5_cropped_half"
BASE_DIR = r"D:\WORK\Periocular-Recognition\images\custom"

def get_data(dir_path):
    """
    Accepts root path to the data folder and returns images and labels
    """
    x,y = [],[]
    
    try:
        dirs = [f for f in listdir(dir_path) if isdir(join(dir_path, f))]
        
        for d in tqdm(dirs):
            current_dir = join(dir_path, d)
            files = [f for f in listdir(current_dir) if isfile(join(current_dir, f))]
            x += [cv2.imread(join(current_dir, f)) for f in files]#
            y += [d]*len(files)
    except Exception as e:
        print(f"[ERROR] get_data : {e}")
        
    return x, y


images, labels = get_data(BASE_DIR)

facenet = Facenet()
features = facenet.get_embeddings(images, verbose=1)

fn = "data\\hface_office_features.pkl"
with open(fn, "wb") as fl:
    pickle.dump({"features": features, "labels": labels}, fl)

print(f"{fn} saved...")