from flask import Flask, render_template, request,redirect, url_for
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

# Load the trained model and other necessary imports
model = load_model('diabetes_model.h5')

# Create a new StandardScaler instance
scaler = StandardScaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = float(request.form['age'])

        # Preprocess input data (scaling)
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        input_data_scaled = scaler.fit_transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(input_data_scaled)
        predicted_class = 1 if prediction[0][0] > 0.4 else 0

        prediction_result = "Positive" if predicted_class == 1 else "Negative"

        return redirect(url_for('show_result', prediction_result=prediction_result,
                                pregnancies=pregnancies, glucose=glucose, blood_pressure=blood_pressure,
                                skin_thickness=skin_thickness, insulin=insulin, bmi=bmi,
                                diabetes_pedigree=diabetes_pedigree, age=age))



@app.route('/show_result/<prediction_result>')
def show_result(prediction_result):
    return render_template('result.html', prediction_result=prediction_result,
                           pregnancies=request.args.get('pregnancies'),
                           glucose=request.args.get('glucose'),
                           blood_pressure=request.args.get('blood_pressure'),
                           skin_thickness=request.args.get('skin_thickness'),
                           insulin=request.args.get('insulin'),
                           bmi=request.args.get('bmi'),
                           diabetes_pedigree=request.args.get('diabetes_pedigree'),
                           age=request.args.get('age'))


if __name__ == '__main__':
    app.run(debug=True)
