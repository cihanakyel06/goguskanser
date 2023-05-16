import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  input_features = [float(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['perimeter_mean', 'area_mean',	'area_se',	'perimeter_worst',	'area_worst']

  df = pd.DataFrame(features_value, columns=features_name)
  output = model.predict(df)

  if output == 'M':
      res_val = "Kanser"
  else:
      res_val = "Normal"


  return render_template('index.html', prediction_text='Hasta {}'.format(res_val))

if __name__ == "__main__":
  app.run()
