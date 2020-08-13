import os, pickle
import numpy as np

# F1 = "data\\leye_features1.pkl"
# F2 = "data\\leye_features2.pkl"
# F3 = "data\\reye_features1.pkl"
# F4 = "data\\reye_features2.pkl"
F1 = "data\\hface_features.pkl"
F2 = "data\\hface_office_features.pkl"

FILES = [F1, F2]

features, labels = None, None


for file1 in FILES:
    with open(file1, "rb") as f1:
        data = pickle.load(f1)
        if features is None:
            features = data["features"]
            labels = data["labels"]
        else:
            features = np.append(features, data["features"], axis=0)
            labels = labels + data["labels"]


print(features.shape)
print(len(labels))


fn = "data\\merged_face_features.pkl"

with open(fn, "wb") as fl:
    pickle.dump({"features": features, "labels": labels}, fl)

print(f"{fn} saved...")
