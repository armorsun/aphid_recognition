import numpy as np
import scipy.misc
import os
from keras.models import model_from_json

def load_imgs():
    folder_path=os.getcwd()+'/trainingData/test/aphids/'
    print(folder_path)
    files=os.listdir(folder_path)
    img_names=[]

    for filename in files:
        path=os.path.join(folder_path,filename)
        img_names.append(path)

    imgs=[np.transpose(scipy.misc.imread(img_name),(2,0,1)).astype('float32') for img_name in img_names]
    return np.array(imgs)/255

def load_model(model_def_fname,model_weights_fname):
    model=model_from_json(open(model_def_fname).read())
    model.load_weights(model_weights_fname)
    return model

if __name__ == '__main__':
    imgs=load_imgs()
    model=load_model('aphid_classifier_model/model_6.json','aphid_classifier_model/6.h5')
    predictions = model.predict_classes(imgs)
    print imgs.shape
    print predictions
    #print imgs[0].shape
