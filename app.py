from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Initialize Flask app
app = Flask(__name__)

# Load dataset from CSV
df = pd.read_csv("new_dataset.csv")

# Encoding the categorical data
le_outlook = LabelEncoder()
le_temp = LabelEncoder()
le_humidity = LabelEncoder()
le_windy = LabelEncoder()

df['outlook_n'] = le_outlook.fit_transform(df['Outlook'])
df['temp_n'] = le_temp.fit_transform(df['Temp'])
df['humidity_n'] = le_humidity.fit_transform(df['Humidity'])
df['windy_n'] = le_windy.fit_transform(df['Windy'])

# Preparing inputs and target
inputs = df[['outlook_n', 'temp_n', 'humidity_n', 'windy_n']]
target = df['Play']

# Naive Bayes model
model = GaussianNB()
model.fit(inputs, target)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    outlook = request.form['outlook']
    temp = request.form['temp']
    humidity = request.form['humidity']
    windy = request.form['windy']

    outlook_n = le_outlook.transform([outlook])[0]
    temp_n = le_temp.transform([temp])[0]
    humidity_n = le_humidity.transform([humidity])[0]
    windy_n = le_windy.transform([windy])[0]

    prediction = model.predict([[outlook_n, temp_n, humidity_n, windy_n]])

    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
