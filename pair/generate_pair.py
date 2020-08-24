import os
import random
import argparse
import sys
from tqdm import tqdm

class GeneratePairs:
    """
    Generate the pairs.txt file for applying "validate on LFW" on your own datasets.
    """

    def __init__(self, args):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = args.data_dir
        self.pairs_filepath = args.saved_dir + 'pairs.txt'
        self.repeat_times = int(args.repeat_times)
        self.img_ext = '.png'

    def _join_name(self, nmlist):
        name = nmlist[0]
        for ix in range(1, len(nmlist), 1):
            name += "_" + nmlist[ix]
        return name

    def generate(self):
        # The repeate times. You can edit this number by yourself
        folder_number = self.get_folder_numbers()

        # This step will generate the hearder for pair_list.txt, which contains
        # the number of classes and the repeate times of generate the pair
        if not os.path.exists(self.pairs_filepath):
            with open(self.pairs_filepath,"a") as f:
                f.write(str(self.repeat_times) + "\t" + str(folder_number) + "\n")
        for i in range(self.repeat_times):
            print(i)
            self._generate_matches_pairs()
            self._generate_mismatches_pairs()

    def get_folder_numbers(self):
        count = 0
        for folder in os.listdir(self.data_dir):
            if os.path.isdir(self.data_dir + folder):
                count += 1
        return count

    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        for name in tqdm(os.listdir(self.data_dir)):
            if name == ".DS_Store" or name[-3:] == 'txt':
                continue

            a = []
            for file in os.listdir(self.data_dir + name):
                if file == ".DS_Store":
                    continue
                a.append(file)

            with open(self.pairs_filepath, "a") as f:
                temp = random.choice(a).split("_") # This line may vary depending on how your images are named.
                w = self._join_name(temp[0:-1])

                pos = random.choice(a).split("_")
                neg = random.choice(a).split("_")
                pindex1, nindex1 = 2, 2

                if len(pos) == 2:   pindex1 = 1
                elif len(pos) == 4: pindex1 = 3
                elif len(pos) == 5: pindex1 = 4
                elif len(pos) == 6: pindex1 = 5

                if len(neg) == 2:   nindex1 = 1
                elif len(neg) == 4: nindex1 = 3
                elif len(neg) == 5: nindex1 = 4
                elif len(neg) == 6: nindex1 = 5


                l = pos[pindex1].lstrip("0").rstrip(self.img_ext)
                r = neg[nindex1].lstrip("0").rstrip(self.img_ext)
                f.write(w + "\t" + l + "\t" + r + "\n")


    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """

        print()

        for i, name in tqdm(enumerate(os.listdir(self.data_dir))):
            if name == ".DS_Store" or name[-3:] == 'txt':
                continue

            remaining = os.listdir(self.data_dir)

            del remaining[i]
            remaining_remove_txt = remaining[:]
            for item in remaining:
                if item[-3:] == 'txt':
                    remaining_remove_txt.remove(item)

            remaining = remaining_remove_txt

            other_dir = random.choice(remaining)
            with open(self.pairs_filepath, "a") as f:
                    file1 = random.choice(os.listdir(self.data_dir + name))
                    file2 = random.choice(os.listdir(self.data_dir + other_dir))

                    msplit1, msplit2 = file1.split("_"), file2.split("_")
                    index1, index2 = 2, 2

                    if len(msplit1) == 2:   index1 = 1
                    elif len(msplit1) == 4: index1 = 3
                    elif len(msplit1) == 5: index1 = 4
                    elif len(msplit1) == 6: index1 = 5

                    if len(msplit2) == 2:   index2 = 1
                    elif len(msplit2) == 4: index2 = 3
                    elif len(msplit2) == 5: index2 = 4
                    elif len(msplit2) == 6: index2 = 5


                    f.write(name + "\t" + msplit1[index1].lstrip("0").rstrip(self.img_ext) \
                     + "\t" + other_dir + "\t" + msplit2[index2].lstrip("0").rstrip(self.img_ext) + "\n")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with aligned images.')
    parser.add_argument('saved_dir', type=str, help='Directory to save pairs.')
    parser.add_argument('--repeat_times', type=str, help='Repeat times to generate pairs', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    generatePairs = GeneratePairs(parse_arguments(sys.argv[1:]))
    generatePairs.generate()