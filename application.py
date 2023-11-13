from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__, template_folder='templates')
car=pd.read_csv("Cleaned Car.csv")

model= pickle.load(open('LinearRegModel3.pkl','rb'))

@app.route('/')
def index():
    companies= sorted(car['company'].unique())
    car_models= sorted(car['name'].unique())
    year= sorted(car['year'].unique())
    fuel_type= sorted(car['fuel_type'].unique())
    return render_template('index.html',companies=companies, car_models=car_models, years=year, fuel_type=fuel_type)

@app.route('/predict',methods=['POST'])
def predict():
    company= request.form.get('company')
    car_models=request.form.get('car_model')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel')
    kms_driven=int(request.form.get('km_driven'))
   

    prediction = model.predict(pd.DataFrame([[car_models, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))


    
    return str(round(prediction[0], 2))


if __name__ =='__main__':
    app.run(debug=True)