from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization

from os import path


class YapaySinirAgi:

    model = None

    def __init__(self, model_path =None):

        if model_path is not None:
            if path.isdir(model_path):
                self.yukle(model_path)
                return
            else:
                raise "Hata, model dosyası yok"


        self.model = Sequential()
        self.model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
        self.model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
        self.model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization())    
        self.model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dense(512,activation="relu"))
        self.model.add(Dense(10,activation="softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
    
    def egit(self, egitimVerisi, devir = 50):
        self.model.fit(egitimVerisi.X, egitimVerisi.y, epochs=devir, batch_size=64)
        _,acc = self.model.evaluate(egitimVerisi.X,egitimVerisi.y)
        print("Eğitim tamamlandı, Doğruluk:", str(acc))
    
    def kaydet(self, dosyaAd):
        save_model(self.model,dosyaAd)
    
    def yukle(self, dosyaKonum):
        self.model = load_model(dosyaKonum)
        print("Model Yüklendi !")
    
    def tahminEt(self,vector):
        tahmin = self.model.predict(vector)
        return tahmin
    


