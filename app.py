
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import os
app = Flask(__name__)

# Eğitilmiş modeli yükle
model = load_model("kalp_krizi_modeli.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        kilo = float(request.form['Kilo'])
        boy = float(request.form['Boy']) / 100  # cm cinsinden gelen boyu metreye çevir
        bmi = kilo / (boy ** 2)  # BMI hesaplama formülü
        # Kullanıcıdan gelen verileri al
        input_features = [
            bmi,
            int(request.form['Sigara']),
            int(request.form['Alkol']),
            int(request.form['Felc']),
            int(request.form['YururkenZorlanma']),
            int(request.form['Cinsiyet']),
            int(request.form['FizikselAktivite']),
            int(request.form['Astim']),
            int(request.form['BobrekHast']),
            float(request.form['UykuSuresi']),
            int(request.form['Diabet']),
            int(request.form['GenetikVarMi']),
            int(request.form['Yas40'])
        ]
        
        # Veriyi numpy dizisine çevir
        input_data = np.array([input_features])
        
        # Modelden tahmin al
        prediction = model.predict(input_data)[0][0]
        
        # Tahmini yüzde formatına çevir
        prediction_percentage = round(prediction * 100, 2)
        
        # Sonucu yüzde olarak göster
        result = f"Kalp krizi geçirme ihtimaliniz: %{prediction_percentage}"
        
        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Hata: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
