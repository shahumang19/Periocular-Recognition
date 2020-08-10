# D:\WORK\Periocular-Recognition\images\ubipr\UBIPeriocular
import os, shutil
from tqdm import tqdm

BASE_DIR = r"D:\WORK\Periocular-Recognition\images\ubipr\UBIPeriocular"
OUT1 = r"D:\WORK\Periocular-Recognition\images\ubipr\left"
OUT2 = r"D:\WORK\Periocular-Recognition\images\ubipr\right"

files = [f for f in os.listdir(BASE_DIR) if f.split(".")[1] == "jpg"]

print(len(files))

for f in tqdm(files):
    out = OUT1
    if int(f.split("_")[0][1:]) % 2 == 0:
        out = OUT2
    # print(f, int(f.split("_")[0][1:]),out)
    # break

    shutil.copyfile(os.path.join(BASE_DIR, f), os.path.join(out, f))