from flask import Flask, render_template, request
import pickle
import pandas as pd

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__, template_folder='template')

features = ['blood_glucose_level', 'HbA1c_level', 'age', 'bmi']

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = [[data1, data2, data3, data4]]
    arr = pd.DataFrame(arr, columns=features)
    pred = model.predict(arr)
    return render_template('after.html', data=pred)
    
if __name__ == "__main__":
    app.run(debug=True)
