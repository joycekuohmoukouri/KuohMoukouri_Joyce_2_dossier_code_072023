import pandas as pd
import numpy as np
import MyModule_p7
import joblib
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from flask import Flask, request, jsonify,  send_file, render_template, redirect
from projet7package.frequency_encode import frequency_encode
import json
import shap
#### Thème de la page
## Les données globales _________________________________
cols = [#'GENRE',
                     'PROPRIETAIRE', 'NBRE_ENFANT',
                     'ANCIENNETE_CREDIT', 'CHARGES_ANNUEL', 'REVENUS_TOT',
                     #'MONTANT_CREDIT',
                    'RATIO_CREDIT_REVENU',
                     'OCCUPATION', 'CC_RATIO_CREDIT', 'NIVEAU_ETUDE', 'AGE',
                     'ANCIENNETE_EMPLOI', 'SCORE_REGION', 'HEURE_APP', 'SECTEUR_ACTIVITE',
                     'SCORE_2_EXT',
                     #'RATIO_ENDETT(%)',
                     'NBRE_CONTRAT_ACTIFS','NBRE_J_RETARD', 'POS_PROGRESS_MAX_MIN',
                     'CC_NOMBRE_RETRAIT_MOYEN', 'CB_SOMME_DUES_RETARD']
df = pd.read_csv('client_test_db.csv', usecols = cols)
# Chgt du pre-processing + modèle
loaded_preprocess = MyModule_p7.preprocess_model()
classification_model = joblib.load('LightGBM_bestmodel.pkl')
df_pp = loaded_preprocess.transform(df)
sv_total, df_pp = MyModule_p7.feat_local(df_pp)

# Create an iframe to embed the welcome page
welcome_url = "https://sleepy-waters-17464-030b06eb8dbd.herokuapp.com/"
st.components.v1.iframe(welcome_url, height=600, scrolling=True)
#------------------------------------------------------------
#Les données propre au client  _________________________________
# Read data from the JSON file
api_url = 'https://sleepy-waters-17464-030b06eb8dbd.herokuapp.com/Dashboard_st/'
response= requests.get(api_url)
if response.status_code == 200:
    data = response.json()
else:
    st.error("Failed to fetch data from the API.")

# Extraction des données utiles
client_id = data.get("client_id")
score = data.get("score")
shap_values = data.get("feat_imp")
client_data = data.get("client_data")

#Conversion des dictionnaires en dataframes
client_df = pd.DataFrame(client_data)
sv_df = pd.DataFrame(shap_values)

# Sort les valeurs des contributions en ordre descendant
sv_df_class0 = sv_df.sort_values('Class_0', ascending=False)
sv_df_class1 = sv_df.sort_values('Class_1', ascending=False)
ls_features_0 = sv_df_class0[:4]['index'].tolist()
ls_features_1 = sv_df_class1[:4]['index'].tolist()
print(sv_df.head())

if score >= 60:
    gauge_color = '#52a676'
elif score >= 50:
    gauge_color = 'yellow'
elif score  >= 40:
    gauge_color = 'orange'
else:
    gauge_color = '#F08080'


print(gauge_color)

# Create two equally sized columns
#col1, col2 = st.columns(2,gap = "large")

# In the first column, create two divisions
# Read the custom CSS file
with open("./static/css/styles_dashboard.css", "r") as css_file:
    custom_css = css_file.read()


st.markdown("<h1 style='text-align: center;'>Votre score</h1>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center;'>N° d'identifiant: {client_id}</div>", unsafe_allow_html=True)
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Score crédit"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': gauge_color},
           'threshold': {
               'line': {'color': "#52a676", 'width': 4},
               'value': 60}
           }
))

st.plotly_chart(fig)
menu_option = st.sidebar.selectbox("Pilotage", ["Points forts du dossier", "Points faibles du dossier"])



if menu_option == "Points forts du dossier":
    selected_feature = st.selectbox("Sélectionnez un point fort", ls_features_0)
    col1, col2 = st.columns(2)
    with col1:
        # Division 1    
        # Create a Plotly bar chart for Class_0
        fig_class0 = px.bar(sv_df_class0[:4], y='index', x='Class_0', title='Points forts',  orientation='h', width=350, height=350)
    
        # Customize the layout (if needed)
        fig_class0.update_xaxes(title=None, showticklabels=False)
        fig_class0.update_yaxes(title=None)
    
        fig_class0.update_layout(
                )
        fig_class0.update_traces(marker_color="#52a676")
        client_value = client_df[selected_feature].values[0] # Replace with the actual client's value
        # Display the Plotly chart
        st.plotly_chart(fig_class0)
    with col2:
        fig_distribution = px.histogram(df_pp, x=selected_feature, title=f'Distribution of {selected_feature}', width=350, height=350)
        fig_distribution.update_traces(marker_color="#474e96")
        fig_distribution.add_vline(x=client_value, line_width=3, line_dash="dash", line_color="#8b7ee7")
        
        fig_distribution.add_annotation(
            x=client_value,
            y=0.9,  # Adjust the y-coordinate of the annotation as needed
            text=f"Vous êtes ici",  # Replace with your desired annotation text
            showarrow=True,
            arrowhead=1,
            arrowcolor="#8b7ee7",  # Adjust the arrow color as needed
            arrowwidth=2  # Adjust the arrow width as needed
    )
        
        # Display the Plotly chart with the distribution and vertical line
        st.plotly_chart(fig_distribution)



if menu_option == "Points faibles du dossier":
    selected_feature = st.selectbox("Sélectionnez un point faible", ls_features_1)
    col1, col2 = st.columns(2)
    with col1:
        # Division 2
        # Create a Plotly bar chart for Class_1
        fig_class1 = px.bar(sv_df_class1[:4], y='index', x='Class_1', title='Points faibles',  orientation='h',width=350, height=350)
    
        # Customize the layout (if needed)
        fig_class1.update_xaxes(title=None, showticklabels=False)
        fig_class1.update_yaxes(title=None)
        fig_class1.update_traces(marker_color='#F08080')
    
        st.plotly_chart(fig_class1)
    with col2:
        fig_distribution = px.histogram(df_pp, x=selected_feature, title=f'Distribution \n {selected_feature}', width=350, height=350)
        fig_distribution.update_traces(marker_color="#474e96")
        client_value = client_df[selected_feature].values[0] # Replace with the actual client's value
        fig_distribution.add_vline(x=client_value, line_width=3, line_dash="dash", line_color="#8b7ee7")
        fig_distribution.add_annotation(
            x=client_value,
            y=0.9,  # Adjust the y-coordinate of the annotation as needed
            text=f"Vous êtes ici",  # Replace with your desired annotation text
            showarrow=True,
            arrowhead=1,
            arrowcolor="#8b7ee7",  # Adjust the arrow color as needed
            arrowwidth=2  # Adjust the arrow width as needed
    )
        
        # Display the Plotly chart with the distribution and vertical line
        st.plotly_chart(fig_distribution)













