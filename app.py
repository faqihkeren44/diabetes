from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['name']
    data2 = request.form['age']
    data3 = request.form['blood_glucose_level']
    data4 = request.form['HbA1c_level']
    data5 = request.form['bmi']
    arr = np.array([[data1, data2, data3, data4, data5]])
    pred = model.predict(arr)
    return render_template('after.html', result=pred)


if __name__ == "__main__":
    app.run(debug=True)