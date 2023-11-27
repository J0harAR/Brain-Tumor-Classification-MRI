from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import cv2
import numpy as np
from tensorflow import keras
import base64
app = Flask(__name__)
CORS(app)  # Habilita CORS para todos los dominios

# Carga tu modelo entrenado
model = keras.models.load_model('modelo.h5')

def img_pred(img):
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    p = model.predict(img)
    p = np.argmax(p, axis=1)[0]
    
    if p == 0:
        return 'Glioma Tumor'
    elif p == 1:
        return 'No tumor'
    elif p == 2:
        return 'Meningioma Tumor'
    else:
       return 'Pituitary Tumor'


# Captura de video desde la cámara
video_capture = cv2.VideoCapture(0)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_camera', methods=['GET', 'POST'])
def predict_camera():
    if request.method == 'POST':
        try:
            # Procesar la imagen y devolver la predicción
            data = request.get_json()
            imagen_data_url = data['imagen_data_url'].split(',')[1]
            img = Image.open(io.BytesIO(base64.b64decode(imagen_data_url)))
            prediction = img_pred(img)
            return jsonify({'prediction': prediction})

        except Exception as e:
            print('Error:', str(e))
            return jsonify({'error': 'Error en la predicción'})
    return jsonify({'error': 'Método no permitido'})


@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        imagen_data_url = data['imagen_data_url'].split(',')[1]
        img = Image.open(io.BytesIO(base64.b64decode(imagen_data_url)))
        prediction = img_pred(img)
        return jsonify({'prediction': prediction})

    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': 'Error en la predicción desde la imagen'})



if __name__ == '__main__':
    app.run(debug=True)
