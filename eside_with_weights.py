# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:03:06 2020

@author: Dr_Wajid (wajidarshad@gmail.com)
"""

import numpy as np
from sklearn import preprocessing
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input

def extract_transfer_learning_features(img_path):
    model = DenseNet121(weights='imagenet', include_top=False)
    img = image.load_img(img_path, target_size=(331, 331))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x).flatten()
    return features
def standarized_normalized_e_vs_o(features, norm='yes'):
    features=(features-np.load('trained_models/mean_e-fetida_vs_others.npy'))/(np.load('trained_models/std_e-fetida_vs_others.npy')+0.0001)
    #features=preprocessing.scale(features)
    if norm=='yes':
        features=preprocessing.normalize(features)
    return features
def apply_eside(image_path):
    feats=extract_transfer_learning_features(image_path)
    trained_model_e_vs_o_w=np.load('trained_models/weights/weight_vector_SVM_densent_e-fetida_vs_others.npy')
    trained_model_e_vs_o_b=np.load('trained_models/weights/bias_SVM_densent_e-fetida_vs_others.npy')
    
    if np.dot(trained_model_e_vs_o_w[0],standarized_normalized_e_vs_o([feats])[0])+trained_model_e_vs_o_b[0]<=0:
        print("E. fetida")
    else:
        print("Some Other Species")
 
if __name__ == "__main__":
    apply_eside('input_image.jpg')
