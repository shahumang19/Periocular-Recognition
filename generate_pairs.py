#! encoding: utf-8

import os
import random

class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.
    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """

    def __init__(self, data_dir, pairs_filepath, img_ext):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = data_dir
        self.pairs_filepath = pairs_filepath
        self.img_ext = img_ext


    def generate(self):
        self._generate_matches_pairs()
        self._generate_mismatches_pairs()


    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        for name in os.listdir(self.data_dir):
            if name == ".DS_Store":
                continue

            a = []
            for file in os.listdir(self.data_dir+ "\\" +name):
                if file == ".DS_Store":
                    continue
                a.append(file)

            with open(self.pairs_filepath, "a") as f:
                for i in range(3):
                    temp = random.choice(a).split("_") # This line may vary depending on how your images are named.
                    w = temp[0:-1]
                    pos = random.choice(a).split("_")
                    neg = random.choice(a).split("_")
                    pindex, nindex = 2, 2

                    if len(pos) == 2:
                        pindex = 1
                    elif len(pos) == 4:
                        pindex = 3
                    elif len(pos) == 5:
                        pindex = 4

                    if len(neg) == 2:
                        nindex = 1
                    elif len(neg) == 4:
                        nindex = 3
                    elif len(neg) == 5:
                        nindex = 4

                    l = pos[pindex].lstrip("0").rstrip(self.img_ext)
                    r = neg[nindex].lstrip("0").rstrip(self.img_ext)
                    f.write(w + "\t" + l + "\t" + r + "\n")


    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        for i, name in enumerate(os.listdir(self.data_dir)):
            if name == ".DS_Store":
                continue

            remaining = os.listdir(self.data_dir)
            remaining = [f_n for f_n in remaining if f_n != ".DS_Store"]
            # del remaining[i] # deletes the file from the list, so that it is not chosen again
            other_dir = random.choice(remaining)
            with open(self.pairs_filepath, "a") as f:
                for i in range(3):
                    file1 = random.choice(os.listdir(self.data_dir+ "\\" +name))
                    file2 = random.choice(os.listdir(self.data_dir+ "\\" +other_dir))
                    msplit = file1.split("_")
                    index = 2

                    if len(msplit) == 2:
                        index = 1
                    elif len(msplit) == 4:
                        index = 3
                    elif len(msplit) == 5:
                        index = 4

                    f.write(name + "\t" + msplit[index].lstrip("0").rstrip(self.img_ext) + "\t")
                f.write("\n")


if __name__ == '__main__':
    data_dir = r"D:\WORK\Periocular-Recognition\images\lfw\lfw_cropped"
    pairs_filepath = "images\\pairs.txt"
    img_ext = ".jpg"
    generatePairs = GeneratePairs(data_dir, pairs_filepath, img_ext)
    generatePairs.generate()