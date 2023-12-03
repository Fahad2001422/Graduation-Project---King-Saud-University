import os
import uuid
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
app = Flask(__name__)

# Load your trained models with relative paths
cancer_model = tf.keras.models.load_model("./my_CancerModel.h5")
pneumonia_model = tf.keras.models.load_model("./my_PneumoniaModel.h5")

@app.route('/')
@app.route('/home')
def home():
    return render_template('MainPage.html')

@app.route('/about')
def about():
    return render_template('AboutPage.html')

@app.route('/services')
def services():
    return render_template('ServicesPage.html')

@app.route('/login')
def login():
    return render_template('LoginPage.html')

@app.route('/signup')
def signup():
    return render_template('Sign-upPage.html')

@app.route('/account')
def account():
    return render_template('AccountPage.html')

@app.route('/static/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file part in the request.'
        }), 400

    file = request.files['file']

    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        filename = str(uuid.uuid4()) + ".png"
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        try:
            image = tf.keras.preprocessing.image.load_img(file_path, target_size=(256, 256))
            image = tf.keras.preprocessing.image.img_to_array(image)

            # Reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

            # Prepare the image for the VGG model
            image = tf.keras.applications.vgg16.preprocess_input(image)

            cancer_prediction = cancer_model.predict(image)
            pneumonia_prediction = pneumonia_model.predict(image)

            # Check if the probability for each disease is too high to show the appropriate response
            healthy_result = dict()
            cancer_result = dict()
            pneumonia_result = dict()

            cancer_result["prediction"] = cancer_prediction[0]
            cancer_result["text"] = ""
            cancer_result["addition"] = "\n"
            cancer_result["hyperlink"] = ""
            if cancer_result["prediction"] > 0:
                cancer_result["text"] = "The patient has a probability of {0}% for lung cancer.\n"\
                    .format(cancer_result["prediction"])
                if cancer_result["prediction"] >= 0.6:
                    cancer_result["addition"] = "{0}% is a huge risk; the patient must be treated in immediate time.\n"\
                        .format(cancer_result["prediction"])
                elif cancer_result["prediction"] < 0.6 and cancer_result["prediction"] >= 0.5:
                    cancer_result["addition"] = "{0}% is a moderate risk. It might be crucial to treat the patient soon.\n"\
                    .format(cancer_result["prediction"])
                else:
                    cancer_result["addition"] = "{0}% is a mild risk. It is advised for the patient to be treated.\n"\
                    .format(cancer_result["prediction"])
                cancer_result["hyperlink"] = """You can give this website to the patient for easier treatment 
                and possibly cheaper healthcare: """


            pneumonia_result["prediction"] = pneumonia_prediction[0]
            pneumonia_result["text"] = ""
            pneumonia_result["addition"] = "\n"
            pneumonia_result["hyperlink"] = ""
            if pneumonia_result["prediction"] > 0:
                pneumonia_result["text"] = "The patient has a probability of {0} for pneumonia.\n"\
                    .format(pneumonia_result["prediction"])
                if pneumonia_result["prediction"] >= 0.6:
                    pneumonia_result["addition"] = "{0}% is a huge risk, the patient must be treated in immediate time.\n"\
                        .format(pneumonia_result["prediction"])
                elif pneumonia_result["prediction"] < 0.6 and pneumonia_result["prediction"] >= 0.5:
                    pneumonia_result["addition"] = "{0}% is a moderate risk. It might be crucial to treat the patient soon.\n" \
                        .format(pneumonia_result["prediction"])
                else:
                    pneumonia_result["addition"] = "{0}% is a mild risk. It is advised for the patient to be treated.\n" \
                        .format(pneumonia_result["prediction"])
                pneumonia_result["hyperlink"] = """You can give this website to the patient for easier treatment 
                    and possibly cheaper healthcare: """


            healthy_result["prediction"] = 0
            healthy_result["text"] = healthy_result["addition"] = healthy_result["hyperlink"] = ""
            if pneumonia_result["prediction"] <= 0.1 and cancer_result["prediction"] <= 0.1:
                healthy_result["prediction"] = 1 - pneumonia_result["prediction"]
                healthy_result["text"] = """The patient has a slight probability of {0}% for lung cancer and
                {1}% for pneumonia.""".format(cancer_result["prediction"], pneumonia_result["prediction"])
                healthy_result["addition"] = "\n"
                healthy_result["hyperlink"] = ""


            # Delete the temporary image file
            os.remove(file_path)

            # Show the most likely predictions on the website
            results_as_json = dict()
            if len(cancer_result["text"]) != 0:
                results_as_json["Cancer prediction"] = cancer_result["text"] + cancer_result["addition"] \
                + cancer_result["hyperlink"]
            if len(pneumonia_result["text"]) != 0:
                results_as_json["Pneumonia prediction"] = pneumonia_result["text"] + pneumonia_result["addition"] \
                + pneumonia_result["hyperlink"]
            if len(healthy_result["text"]) != 0:
                results_as_json["Healthy prediction"] = healthy_result["text"] + healthy_result["addition"]

            # Convert the prediction data into JSON format
            return jsonify(results_as_json)

        except Exception as e:
            return jsonify({
                'error': str(e)
            }), 500

    return jsonify({
        'error': 'File not allowed.'
    }), 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=int("80"))