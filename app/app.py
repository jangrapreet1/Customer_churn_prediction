from flask import Flask, render_template,request
import numpy as np
import pandas as pd
from joblib import load
import uuid

import plotly.express as px
import plotly.graph_objects as go

import kaleido

def make_picture(training_data_filename, model, new_inp_np_arr,output_file):
    data = pd.read_pickle(training_data_filename)
    ages = data["Age"]
    data = data[ages>0]
    ages = data["Age"]
    heights = data["Height"]
    x_new = np.array(list(range(19))).reshape(19,1)
    y_preds = model.predict(x_new)

    fig = px.scatter(x=ages, y=heights, title="Height vs Ages of People", labels = {'x': 'Age in Years',
                                                                                    'y': 'Height of People in Inches'})
    fig.add_trace(go.Scatter(x=x_new.reshape(19), y=y_preds, mode="lines", name="Model",
                            line=dict(color='red', dash='dashdot')))
    
    new_preds = model.predict(new_inp_np_arr)
    
    fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name= 'New Outputs', mode='markers', marker =dict(color='purple',size=10,line=dict(color='purple',width=2))))

    fig.write_image(output_file,engine = "kaleido",width=800,format='svg')

    fig.show()

def floats_string_to_np_arr(floats_str):
    def is_float(s):
        try:
            float(s)
            return True

        except:
            return False
    
    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats),1)


app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str =='GET':
        return render_template('index.html', href = 'static/base_image.svg')
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static/"+ random_string +".svg"
        model = load('model.joblib')
        np_array = floats_string_to_np_arr(text)
        make_picture('AgesAndHeights.pkl',model, np_array, path)
        
        return render_template('index.html', href = path)