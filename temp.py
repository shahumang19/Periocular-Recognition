import os
import shutil
import imagesize
from tqdm import tqdm

BASE_DIR = r"C:\Umang\Periocular-Recognition\images\CASIA\CASIA-ALIGNED-CROPPED"
dirs = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]

rmcount = 0

for ix in tqdm(range(len(dirs))):
    for f in os.listdir(os.path.join(BASE_DIR, dirs[ix])):
        width, height = imagesize.get(f"{BASE_DIR}\\{dirs[ix]}\\{f}")
        if width < 160 or height < 160:
            os.remove(f"{BASE_DIR}\\{dirs[ix]}\\{f}")
            rmcount += 1

print(f"wh < 160 :  {rmcount}")

rmcount = 0
for ix in tqdm(range(len(dirs))):
    files = os.listdir(os.path.join(BASE_DIR, dirs[ix]))
    if len(files) == 0:
        os.remove(os.path.join(BASE_DIR, dirs[ix]))
        rmcount += 1

print(f"empty :  {rmcount}")     
