from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import calendar

encoder = joblib.load('county_encoder.gz')
model1_x_scaler = joblib.load('model1_x_scaler.gz')
mlp = joblib.load('mlp.gz')
lr = joblib.load('model2.gz')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model1', methods=['POST'])
def model1():
    #get values from "gui"
    selected_county = request.form['county']
    numberCrewsInvolved = request.form['crewsInvolved']
    numberPersonnelInvolved = request.form['PersonnelInvolved']
    
    #Encode crews
    encodedCounty = encoder.transform(np.array(selected_county).reshape(-1, 1))

    #Scale Numerical values
    x = pd.DataFrame({'CrewsInvolved': [numberCrewsInvolved], 'PersonnelInvolved': [numberCrewsInvolved]})
    x_rescaled = pd.DataFrame(model1_x_scaler.transform(x), columns=x.columns)

    #Predict with model
    x_final = np.hstack((encodedCounty.toarray(), x_rescaled))
    result = mlp.predict(x_final)

    return f"The selected county is: {selected_county}. The number of crews involved is: {numberCrewsInvolved}. The number of personnel involved is: {numberPersonnelInvolved}. Result: {result[0]} injuries"

def model_prediction(month, prev_count, prev_temp):
  lm = lr[lr['Month'] == float(month)]
  x = pd.DataFrame({"t1": [prev_temp], "n1": [prev_count]})
  x_rescaled = pd.DataFrame(lm["x_scaler"][0].transform(x), columns=x.columns)

  pred_scaled = lm["lr"][0].predict(x_rescaled)
  pred = lm["y_scaler"][0].inverse_transform(pred_scaled)[0][0]
  print(x)
  print(x_rescaled)
  print(pred)
  return (round(pred))

@app.route('/model2', methods=['POST'])
def model2():
    selectedMonth = request.form['months']
    numFires = request.form['firesLastMonth']
    selectedTemp = request.form['lastTemperature']

    result = model_prediction(selectedMonth, numFires, selectedTemp)
    return f"The selected month is: {selectedMonth}. The selected temperature is: {selectedTemp}. The selected number of fires is: {numFires}. The predicted result is: {result}"

if __name__ == '__main__':
    app.run()