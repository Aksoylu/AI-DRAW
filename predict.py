import pandas as pd
import numpy as np
import cv2
from nnet import YapaySinirAgi

def resimOku(path):
    
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) / 255
    img = cv2.resize(img,(28,28) , interpolation = cv2.INTER_AREA)
    img = img.reshape(1,28, 28, 1)
    return img



okunanResim = resimOku("Images/resim_7.png")

yapayzeka = YapaySinirAgi()
yapayzeka.yukle("ytu_egitilmis_model_50")

tahminSonuc = yapayzeka.tahminEt(okunanResim)
tahminSonuc = np.argmax(tahminSonuc)
print("Tahmin Sonucu => [",tahminSonuc,"]")
