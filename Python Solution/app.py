from flask import Flask, render_template,request,send_file

import pickle
import pandas as pd
import numpy as np
from geopy.distance import geodesic

model=pickle.load(open('Cab_Fare_Prediction_model.model','rb'))
scaler=pickle.load(open('scaler.model','rb'))

app = Flask(__name__)

def distance(dataset):
    geodesic_dist=[]
    
    for i in range(len(dataset)):
        pickup = (dataset.pickup_latitude.iloc[i], dataset.pickup_longitude.iloc[i])
        dropoff = (dataset.dropoff_latitude.iloc[i], dataset.dropoff_longitude.iloc[i])
        geodesic_dist.append(abs(round(geodesic(pickup, dropoff).miles,2)))
        
    dataset['distance']=geodesic_dist
    dataset.drop(columns=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],inplace=True)

    return dataset

def time_features(dataset):
    dataset['year']=pd.DatetimeIndex(dataset.pickup_datetime).year
    dataset['month']=pd.DatetimeIndex(dataset.pickup_datetime).month
    dataset['week_day']=pd.DatetimeIndex(dataset.pickup_datetime).weekday
    dataset['hour']=pd.DatetimeIndex(dataset.pickup_datetime).hour
       
    dataset.drop(columns=['pickup_datetime'],inplace=True)
    
    return dataset

def cab_type(dataset):
    dataset['cab_type']=[0 if i<4 else 1 for i in dataset.passenger_count ]
    dataset.drop(columns=['passenger_count'],inplace=True)
    
    return dataset

def predict(test_data):
    df=test_data.copy()
    test_data=distance(test_data)
    test_data=time_features(test_data)
    test_data=cab_type(test_data)
    
    col=test_data.columns
    test_data[col]=scaler.transform(test_data[col])
    pred=model.predict(test_data)
    
    test_data=df
    test_data['fare_amount']=pred
    test_data.fare_amount=test_data.fare_amount.transform(lambda x: round(x,2) )
    return test_data
    

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def upload_predict():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        shape = df.shape
        df=predict(df)
        df.to_csv('predictions.csv',index=False)
        return render_template('predict.html', shape=shape,d=df.values)
    else:
        return render_template('predict.html')
        
@app.route('/csv')  
def download_csv():  
    return send_file('predictions.csv',
                     mimetype='text/csv',
                     attachment_filename='predictions.csv',
                     as_attachment=True)

    

if __name__=='__main__':
    app.run(debug=True) 


