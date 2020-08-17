# D:\WORK\Periocular-Recognition\images\ubipr\UBIPeriocular
import cv2
import os, shutil
from tqdm import tqdm
from os import listdir, mkdir
from os.path import isfile, isdir, join, exists
from FaceDetectionDlib import FaceDetectAndAlign

BASE_DIR = r"D:\WORK\Periocular-Recognition\images\lfw\lfw-deepfunneled"
OUT1 = r"D:\WORK\Periocular-Recognition\images\lfw\lfw_cropped"


def crop_data(base_path, out_path):
    """
    Accepts root path to the data folder and returns images and labels
    """
    
    # try:
    fd = FaceDetectAndAlign(desiredFaceWidth=128, align=False)
    dirs = [f for f in listdir(base_path) if isdir(join(base_path, f))]
    
    for d in tqdm(dirs):
        current_dir = join(base_path, d)
        out_dir = join(out_path, d)
        if not exists(out_dir):
            mkdir(out_dir)
            for f in [f for f in listdir(current_dir) if isfile(join(current_dir, f))]:
                img = cv2.imread(join(current_dir, f))
                fbox = fd.detect_faces(img, biggest=True)
                if len(fbox) > 0:
                    face = fd.extract_faces(img, fbox)[0]
                    if face is not None:
                        cv2.imwrite(join(out_dir, f), face)
        else:
            continue
    # except Exception as e:
    #     print(f"[ERROR] get_data : {e}")


crop_data(BASE_DIR, OUT1)



# empty_dir = []

# dirs = [f for f in listdir(OUT1) if isdir(join(OUT1, f))]

# for d in dirs:
#     current_dir = join(OUT1, d)
#     if len(os.listdir(current_dir)) == 0:
#         # os.rmdir(current_dir)
#         print(d)
