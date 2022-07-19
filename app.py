from flask import Flask,render_template,request
import pandas as pd
import numpy as np

import joblib

app = Flask(__name__)
data=pd.read_csv('Cleaned_data.csv')
pipe=joblib.load('housepred.joblib')

@app.route("/")
def index():
    locations=sorted(data['location'].unique())
    return render_template("index.html",locations=locations)

@app.route("/predict",methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqft=request.form.get('sq.ft')
    print(location,bhk,bath,sqft)
    input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','BHK'])
    prediction=pipe.predict(input)[0]
    prediction=np.exp(prediction)
    prediction=int(prediction*1e5)
    return str(prediction)

if __name__ == "__main__":
  app.run(debug=False)