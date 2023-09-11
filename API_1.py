import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from flask import Flask, request, jsonify, send_file, render_template, redirect
import subprocess
import json
import joblib
import MyModule_p7
import shap  # Import the 'shap' module
import os
from projet7package.frequency_encode import frequency_encode

## Création de l'API

app = Flask(__name__, static_url_path='/static')
app.config["DEBUG"] = True

dashboard_data = {}

@app.route('/')
def welcome():
    return render_template('/index.html')

# Define a route for the welcome page
@app.route('/Dashboard/',methods=['GET','POST'])
def Dashboard():
    from projet7package.frequency_encode import frequency_encode
    # Receive client ID from the form submission
    client_id = request.form.get('client_id')
    if client_id is not None:
        # Chargement des données client using your module
        client_data = MyModule_p7.get_client_data(client_id)
        # Transformation
        loaded_preprocess = MyModule_p7.preprocess_model()
        df_client_pp = loaded_preprocess.transform(client_data)
        classification_model = joblib.load('LightGBM_bestmodel.pkl')
        #Prédiction
        prediction = classification_model.predict(df_client_pp)
        proba = classification_model.predict_proba(df_client_pp)
        score = int(round((proba[0][0])*100)) #probabilité complémentaire
        # Feature analysis using SHAP values
        SV, df_client_pp = MyModule_p7.feat_local(df_client_pp)
        # Dataframe sv_df
        sv_df = pd.DataFrame(columns=['Class_0', 'Class_1'], index=df_client_pp.columns)
        sv_df['Class_0'] = SV[0].T
        sv_df['Class_1'] = SV[1].T
        sv_df = sv_df.reset_index()
        sv_df = sv_df.to_dict()
        df_client_pp = df_client_pp.to_dict()
        global dashboard_data
        dashboard_data = {'client_id': client_id,
        'score': score,
        'feat_imp' :sv_df,
        'client_data' : df_client_pp}

        
        return jsonify({'client_id': client_id,
        'score': score,
        'feat_imp' :sv_df,
        'client_data' : df_client_pp})
    else:
        return 'Nope'



@app.route('/Dashboard_st/',methods=['GET'])
def another_endpoint():
    global dashboard_data
    if not dashboard_data:
        return jsonify({'error': 'No data available from the Dashboard.'}), 404
    
    return jsonify(dashboard_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Default to port 5000 if not specified
    app.run(host='0.0.0.0', port=port)



