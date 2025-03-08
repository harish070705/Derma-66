from flask import render_template, jsonify, Flask, redirect, url_for, request
from flask_mail import Mail, Message
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from PIL import Image
import io

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = 'harishkrishnam640@gmail.com'

# Initialize Flask-Mail
mail = Mail(app)

# Define the SKIN_CLASSES dictionary here at the global scope
SKIN_CLASSES = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowens disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

# **Load the model ONCE when the app starts**
from tensorflow.keras.models import load_model

model = load_model("model.h5", compile=False)  # Ignore incompatible optimizer settings
print("✅ Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return render_template('uploaded.html', title='Upload an Image')

        # Read the image from the file storage and convert it to a PIL Image
        image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224))  # Resize to match model input

        # Convert to NumPy array & preprocess
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = img_array.reshape((1, 224, 224, 3))

        # **Make prediction using the loaded model**
        prediction = model.predict(img_array)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]

        return render_template('uploaded.html', title='Success', predictions=disease, acc=accuracy * 100, img_file=f.filename)

    return render_template('uploaded.html', title='Upload an Image')

@app.route('/send-email', methods=['POST'])
def send_email():
    try:
        data = request.get_json()
        name = data['name']
        email = data['email']

        msg = Message('Thanks for Signing Up!',
                      sender=app.config['MAIL_DEFAULT_SENDER'],
                      recipients=[email])
        msg.body = f"Hi {name},\n\nThank you for contacting our website!"

        mail.send(msg)
        return jsonify({'status': 'success', 'message': 'Email sent successfully!'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
if __name__ == "__main__":
    app.run(debug=True)
