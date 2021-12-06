from logging import debug
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from os import path, remove
import cv2
import numpy as np
from nnet import YapaySinirAgi
import util

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

yapayzeka = YapaySinirAgi()
yapayzeka.yukle("ytu_egitilmis_model_50")

def resimOptimizasyon(resim):
    
    resim = cv2.resize(resim,(28,28) , interpolation = cv2.INTER_AREA)
    resim = cv2.bitwise_not(resim)
    resim = resim / 255
    resim = resim.reshape(1,28, 28, 1)
    
    return resim


@app.route("/")
def main():
    return render_template('root.html')

@app.route("/yapayzeka_analiz", methods=['POST'])
def analiz():
    
    _image = request.files['image_file']
    
    image_name = util.get_random_string(5) + ".png"
    saved_image_path = path.join("cache", image_name)
    _image.save(saved_image_path)

    if path.exists(saved_image_path):
        img = cv2.imread(saved_image_path, cv2.IMREAD_GRAYSCALE)
        remove(saved_image_path)    #flush cached image
        img = resimOptimizasyon(img)
        tahminSonuc = yapayzeka.tahminEt(img)
        tahminSonuc = np.argmax(tahminSonuc)
        print("Tahmin Sonucu => [",tahminSonuc,"]")
        return str(tahminSonuc)
    else:
        return "resim yüklenemedi"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True, threaded=True)
