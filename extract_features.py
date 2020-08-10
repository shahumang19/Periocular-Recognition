from Facenet import Facenet
from tqdm import tqdm
import numpy as np
import cv2
import pickle
import os

# BASE_LEFT = r"D:\WORK\Periocular-Recognition\images\ubipr\left\l2"
BASE_RIGHT = r"D:\WORK\Periocular-Recognition\images\ubipr\right\r2"

# lfiles = os.listdir(BASE_LEFT)
rfiles = os.listdir(BASE_RIGHT)

# print(len(lfiles))
# print(len(rfiles))

facenet = Facenet()

# eye_images = [cv2.imread(os.path.join(BASE_LEFT, f)) for f in tqdm(lfiles)]
# eye_features = facenet.get_embeddings(eye_images, verbose=1)
# labels = [f.split("_")[0] for f in lfiles]

# print(len(labels))
# print(eye_features.shape)
    
# fn = "data\\leye_features2.pkl"
# with open(fn, "wb") as fl:
#     pickle.dump({"features": eye_features, "labels": labels}, fl)

# print(f"{fn} saved...")




labels = []
for f in rfiles:
    c = f.split("_")[0][1:]
    labels.append(f"C{int(c)-1}")

eye_images = [cv2.imread(os.path.join(BASE_RIGHT, f)) for f in tqdm(rfiles)]
eye_features = facenet.get_embeddings(eye_images, verbose=1)

print(len(labels))
print(eye_features.shape)
    
fn = "data\\reye_features2.pkl"
with open(fn, "wb") as fl:
    pickle.dump({"features": eye_features, "labels": labels}, fl)

print(f"{fn} saved...")

