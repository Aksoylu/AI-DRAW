import pandas as pd
import numpy as np
import cv2
from nnet import YapaySinirAgi

def resimOku(path):
    
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = img.flatten()
    return np.array(img)

okunanResim = resimOku("Images/resim_7.png")
okunanResim = okunanResim.reshape(1,-1)

yapayzeka = YapaySinirAgi()
yapayzeka.yukle("ytu_egitilmis_model_150")

tahminSonuc = yapayzeka.tahminEt(okunanResim)
tahminSonuc = np.argmax(tahminSonuc)
print("Tahmin Sonucu => [",tahminSonuc,"]")
