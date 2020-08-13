import cv2, numpy as np
import pickle
from os import listdir
from os.path import isfile, isdir, join
from annoy import AnnoyIndex
from time import time


def read_config():
    file = open('config.txt','r')
    config = file.readline()
    val = eval(config)
    return val



def write_data(data,name):
    """
    Writes data to pickle file.
    name : name of the pickle file
    """
    try:
        with open(name, "wb") as fi:
            pickle.dump(data, fi)
    except Exception as e:
        print(f"[ERROR] write_data : {e}")

        
def read_data(name):
    """
    Reads data from pickle file
    name : name of the pickle file
    """
    try:
        with open(name, "rb") as fi:
            data = pickle.load(fi)
            return data
    except Exception as e:
        print(f"[ERROR] read_data :  {e}")
        


def get_labels(path="data/labels.pkl"):
    """
    read data from pickle file
    """
    labels = read_data(path)
    return labels
    


def get_features(path="data/features.pkl"):
    """
    read features from pickle file
    """
    return read_data(path)


def get_data(dir_path):
    """
    Accepts root path to the data folder and returns images and labels
    """
    x,y = [],[]
    
    try:
        dirs = [f for f in listdir(dir_path) if isdir(join(dir_path, f))]
        
        for d in dirs:
            current_dir = join(dir_path, d)
            files = [f for f in listdir(current_dir) if isfile(join(current_dir, f))]
            x += [cv2.imread(join(current_dir, f)) for f in files]#
            y += [d]*len(files)
    except Exception as e:
        print(f"[ERROR] get_data : {e}")
        
    return x, y
    

def load_images(dir_path):
    """
    Accepts root path to the images folder and returns images
    """
    images = []
    
    try:
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        images = [cv2.imread(join(dir_path, f)) for f in files]
    except Exception as e:
        print(f"[ERROR] load_images : {e}")
    
    return images


def generate_annoyIndex(xp, name, trees=10):
    """
    Generates annoy index for given numpy array and saves the file on disk
    xp : (?,128) dimensional feature array
    name : name of the file
    trees : number of trees for annoy index (more trees more accuracy) default is 10.
    """
    try:
        f = 128
        start = time()

        t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
        for i, feature in enumerate(xp):
            t.add_item(i, feature)
        print("[INFO]Face features added to the Annoy object")

        t.build(trees) # 10 trees
        t.save(name)
        print(f"[INFO] Time taken for index file generation : {str(time() - start)}")
    except Exception as e:
        print(f"[ERROR] generate_annoyIndex : {e}")


def generate_annoyIndex_live(xp, trees=10):
    """
    Generates annoy index for given numpy array and saves the file on disk
    xp : (?,128) dimensional feature array
    name : name of the file
    trees : number of trees for annoy index (more trees more accuracy) default is 10.
    """
    try:
        f = 128
        start = time()

        t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
        for i, feature in enumerate(xp):
            t.add_item(i, feature)
        print("[INFO]Face features added to the Annoy object")

        t.build(trees) # 10 trees
        print(f"[INFO] Time taken for index file generation : {str(time() - start)}")
        
        return t
    except Exception as e:
        print(f"[ERROR] generate_annoyIndex : {e}")

   
def load_index(name):
    """
    Loads annoy index from disk
    name : name of the file
    """
    try:
        index = AnnoyIndex(128, 'euclidean')
        index.load(name)
        return index
    except Exception as e:
        print(f"[ERROR] load_index : {e}")
        
    return None
    
    
def search(query, index, neighbours=3):
    """
    Returns nearest neighbours of the query vector
    query : (128,) dimensional vector
    index : annoy index object
    neighbours : number of nearest neighbours
    """
    
    return index.get_nns_by_vector(query, neighbours, include_distances=True)
  

def get_predictions(face_embeddings, annoy_index, labels, thresh=0.50):
    """
    Returns the predictions of the faces passed as parameter from annoy object
    face_embeddings : face embeddings of multiple faces
    annoy_index : annoy index object
    labels : labels for the faces
    thresh : Distance threshold for comparision between faces (less threshold means strict criteria)
    """
    predictions = []
    
    for ix, face_emb in enumerate(face_embeddings):
        out = search(face_emb, annoy_index)
        name_count = [labels[ind] for ind in out[0]]
        name_count = name_count.count(name_count[0])
        if name_count >= 3:
            index = out[0][0]
            dist = out[1][0]
            
            if dist <= thresh:
                name = labels[index]
                predictions.append((name, dist))
                continue
            
        predictions.append(("Unknown", 0.0))
    return predictions

# def get_predictions(face_embeddings, annoy_index, labels, ids, thresh=0.50):
    # """
    # Returns the predictions of the faces passed as parameter from annoy object
    # face_embeddings : face embeddings of multiple faces
    # annoy_index : annoy index object
    # labels : labels for the faces
    # thresh : Distance threshold for comparision between faces (less threshold means strict criteria)
    # """
    # predictions = []
    
    # for face_emb in face_embeddings:
        # out = search(face_emb, annoy_index)
        # name_count = [labels[ind] for ind in out[0]]
        # name_count = name_count.count(name_count[0])
        # if name_count >= 3:
            # index = out[0][0]
            # dist = out[1][0]
            
            # if dist <= thresh:
                # name = labels[index]
                # id_ = ids[index]
                # predictions.append((name, dist, id_))
                # continue
            
        # predictions.append(("Unknown", 0.0, ""))
    # return predictions

    
def draw_predictions(img,cords, predictions):
    """
    Draws predictions on images
    img : The image on which the predictions will be drawn
    cords : face locations
    predictions : The prediction list containing name and distance of each face location
    """
    img_cp = img.copy()
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1.5
    
    for (x,y,w,h),(name, dist) in zip(cords, predictions):
        text = f"{name} : {dist:.4f}"
        cv2.rectangle(img_cp, (x,y), (x+w, y+h), (0,0,255), int(0.01*img_cp.shape[0]))
        (tw, th) = cv2.getTextSize(text, font, font_scale, thickness=5)[0]
        cv2.rectangle(img_cp, (x,y-th), (x+tw, y), (0,0,0), -1)
        cv2.putText(img_cp, text, (x, y), font, font_scale, (255,255,255), 4)

    return img_cp


def reduce_glare(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = clahe.apply(gray)
    image = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return image