# D:\WORK\Periocular-Recognition\images\ubipr\UBIPeriocular
import cv2, multiprocessing
import os, shutil
from tqdm import tqdm
from os import listdir, mkdir
from os.path import isfile, isdir, join, exists
from FaceDetectionDlib import FaceDetectAndAlign

BASE_DIR = r"D:\WORK\Periocular-Recognition\images\lfw\lfw-deepfunneled"
OUT1 = r"D:\WORK\Periocular-Recognition\images\lfw\lfw_cropped"
PROCESSES = 3



def align_crop_face(fd, current_dir, out_dir):
    if not exists(out_dir):
        mkdir(out_dir)

    for f in [f for f in listdir(current_dir) if isfile(join(current_dir, f))]:
        if not exists(join(out_dir, f)):
            img = cv2.imread(join(current_dir, f))
            fbox = fd.detect_faces(img, biggest=True)
            if len(fbox) > 0:
                faces = fd.extract_faces(img, fbox)
                if len(faces) > 0:
                    # print(faces[0].shape)
                    if faces[0] is not None:
                        if faces[0].shape[0] > 0 and faces[0].shape[1] > 0:
                            cv2.imwrite(join(out_dir, f), faces[0])
                        del faces
        # else:
        #     print(f"[Exists] {join(out_dir, f)}")




def crop_data(base_path, out_path, num_processes):
    """
    Accepts root path to the data folder and returns images and labels
    """
    
    dirs = [f for f in listdir(base_path) if isdir(join(base_path, f))]
    fds = [FaceDetectAndAlign(desiredFaceWidth=512) for _ in range(num_processes)] 

    dir_count, len_dirs = 0, len(dirs)

    while dir_count <=  len_dirs:
        processes = []
        for ix in range(num_processes):
            current_dir = join(base_path, dirs[dir_count])
            out_dir = join(out_path, dirs[dir_count])
            p = multiprocessing.Process(target=align_crop_face, args=(fds[ix], current_dir, out_dir,))
            processes.append(p)
            p.start()
            dir_count += 1

        for p in processes:
            p.join()
        
        print(f"Completed : {dir_count}/{len_dirs}")


if __name__ == "__main__":
    crop_data(BASE_DIR, OUT1, PROCESSES)



    # empty_dir = []

    # dirs = [f for f in listdir(OUT1) if isdir(join(OUT1, f))]

    # for d in dirs:
    #     current_dir = join(OUT1, d)
    #     if len(os.listdir(current_dir)) == 0:
    #         # os.rmdir(current_dir)
    #         print(d)