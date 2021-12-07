#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import seaborn as sns
import os
import requests
import json

# Téléchargement des données
default_dir = os.getcwd()
data = pd.read_csv(os.path.join(default_dir,'data_sampled.csv'))
data_chart = pd.read_csv(os.path.join(default_dir,'data_chart_sampled.csv'))

# Choix du mode de fonctionnement
mode = st.selectbox('Choisissez le mode', options = ['Graphiques','Prediction'], index=1)


# Mode affichage de graphique
if mode ==  'Graphiques' :
    features = st.multiselect("Choisissez deux variables", list(data_chart.columns))
    if len(features) != 2 :
        st.error("Sélectionnez deux variables")
    else :
        st.write("## Graphique bi-varié")
        # Graphique
        chart = sns.jointplot(x=data_chart[features[0]], y=data_chart[features[1]], height=10)
        # Regression linéaire
        sns.regplot(x=data_chart[features[0]], y=data_chart[features[1]], scatter=False, ax=chart.ax_joint)
        st.pyplot(chart)
    st.button("Recommencer")
        
# Mode prédiction
if mode == 'Prediction' :
    profile_ID = st.multiselect("Choisissez un profil", list(data['SK_ID_CURR']), default = 149741)
    if len(profile_ID) != 1 :
        st.error("Sélectionnez un seul profil")
    else :
        profile_ID = str(profile_ID)
        query_str = str('SK_ID_CURR == '+ profile_ID)
        profile_data = data.query(query_str)

        profile_data = profile_data.drop(['SK_ID_CURR','TARGET'], axis = 1)
        request = profile_data.to_json
    
        URL='https://predictionp7.herokuapp.com/predict'
        r = requests.post(URL, json=request)
        st.write(r.json())
    st.button("Recommencer")
 
 
 


# In[5]:


1 !=2


# In[ ]:




