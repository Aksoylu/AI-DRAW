from logging import debug
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from os import path
import cv2
import numpy as np
from nnet import YapaySinirAgi

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

yapayzeka = YapaySinirAgi()
yapayzeka.yukle("ytu_egitilmis_model_20")

def resimOptimizasyon(resim):

    resim = resim.flatten()
    return np.array(resim)


@app.route("/")
def main():
    return render_template('root.html')

@app.route("/yapayzeka_analiz", methods=['POST'])
def analiz():
    
    _image = request.files['image_file']
    
    saved_image_path = path.join("cache", "dosya.jpg")
    _image.save(saved_image_path)

    if path.exists(saved_image_path):

        img = cv2.imread(saved_image_path, cv2.IMREAD_GRAYSCALE)
        yeniBoyut = (28,28)
        img = cv2.resize(img,yeniBoyut)
        img = resimOptimizasyon(img).reshape(1,-1)
        tahminSonuc = yapayzeka.tahminEt(img)
        tahminSonuc = np.argmax(tahminSonuc)
        print("Tahmin Sonucu => [",tahminSonuc,"]")
        return str(tahminSonuc)
    else:
        return "resim yüklenemedi"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True, threaded=True)
