import keras as k, numpy as np
import cv2

class Facenet():
    def __init__(self, path="models\\facenet_keras.h5"):
        """ 
        Takes path where model is stored and returns the loaded model
        path : Path to the facenet model
        """
        try:
            self.model = k.models.load_model(path, compile=False)
            print("[INFO] Facenet model loaded")
        except Exception as e:
            print(f"[ERROR] Error occured while Loading the {path} model : {e}")


    def preprocess(self, f):
        """
        Preprocesses the face image and returns the processed image
        f : Face image
        """
        sf = None
        try:
            sf = cv2.resize(f, (160,160))
            sf = sf.astype("float32")
            mean, std = sf.mean(), sf.std()
            sf = (sf - mean) / std
        except Exception as e:
            print(f"[ERROR] Facenet preprocess :  {e}")
        return sf


    def l2_normalize(self, x, axis=-1, epsilon=1e-10):
        """
        Normalizes the facenet features
        x : (1,128) dimensional 
        """
        try:
            output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
            return output
        except Exception as e:
            print(f"[ERROR] Facenet l2_normalize : {e}")
        return None

    
    def get_embeddings(self, faces, verbose=0):
        """
        Accepts face image and returns 128 dimensional feature vector
        face_img : Face image
        model : facenet model
        """
        face_features = []
        try :
            for ix,face in enumerate(faces):
                s_face = self.preprocess(face)
                s_face = s_face.reshape((-1,160,160,3))
                feature = self.model.predict(s_face)
                nfeature = self.l2_normalize(feature)
                face_features.append(nfeature[0])
                if verbose == 1 : print(f"Processed : {ix+1}/{len(faces)}");
        except Exception as e:
            print(f"[ERROR] Facenet get_embeddings : {e}")
            return None
        
        return np.asarray(face_features)
    
    
if __name__ == "__main__":
    fn = Facenet()
    tt = cv2.imread("images\\IMG-2216.JPG")
    print("------------------------------------------------")
    print(fn.get_embeddings([tt]))