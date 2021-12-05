import pandas as pd
import numpy as np
from nnet import YapaySinirAgi


def category_encode(etiketler, etiketCesit = 10):

    categorized_etiketler = []
    for i in etiketler:
        vector = np.zeros(etiketCesit)
        vector[i] = 1
        categorized_etiketler.append(vector)
    return np.array(categorized_etiketler)



dataset = pd.read_csv("dataset.csv")

y = dataset["label"]
y = category_encode(y)
X = dataset.drop("label", axis = 1)

def egitimVerisi():   pass

egitimVerisi.X = X
egitimVerisi.y = y


yapayzeka = YapaySinirAgi()
yapayzeka.egit(egitimVerisi,20)
yapayzeka.kaydet("ytu_egitilmis_model_20")



