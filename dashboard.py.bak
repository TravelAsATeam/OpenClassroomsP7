#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import seaborn as sns
import os
import requests
import shap
import joblib
st.set_option('deprecation.showPyplotGlobalUse', False)

# Téléchargement des données
default_dir = os.getcwd()
data = pd.read_csv(os.path.join(default_dir,'data_sampled.csv'))
data_chart = pd.read_csv(os.path.join(default_dir,'data_chart_sampled.csv'))
data.reset_index(inplace=True, drop = True)
# Données d'interprétabilité
with open('val_file.pkl', 'rb') as f:
     shap_values = pickle.load(f)


# Calcul de l'interpretabilite
#st.write("Calcul de l'interpretabilite, patientez")
#model = joblib.load('model_rf.pkl')
#explainer = shap.Explainer(model, data.drop(['SK_ID_CURR','TARGET'], axis = 1))
#shap_values = explainer(data.drop(['SK_ID_CURR','TARGET'], axis = 1), check_additivity=False)
#st.write("Calcul terminé")

# Choix du mode de fonctionnement
mode = st.selectbox('Choisissez le mode', options = ['Graphiques','Prediction','Interprétabilité globale'], index=1)


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
        profile_index = profile_data.index[0]
        profile_data = profile_data.drop(['SK_ID_CURR','TARGET'], axis = 1)
        profile_data = profile_data.to_dict(orient='list')
        request = json.dumps(profile_data)
        
        URL='http://travelasateam.pythonanywhere.com/predict/'
        headers = {'Content-Type': 'application/json'}
        
        r = requests.post(URL, headers=headers, data = request, verify=False)
        if r.json()[0]>0.5 :
            st.write('La prediction par machine learning apporte un avis défavorable.')
        else : st.write('La prediction par machine learning apporte un avis favorable.')
        # Interprétabilité locale
        st.write('Le graphique suivant indique les variables ayant le plus contribué à la prédiction et dans quel sens.')
        waterfall = shap.plots.waterfall(shap_values[profile_index], max_display=20)
        st.pyplot(waterfall)
        
# Mode interprétabilité globale
if mode ==  'Interprétabilité globale' :
    st.write('Les graphiques suivants indiquent les variables ayant le plus contribué au modèle.')
    beeswarm = shap.plots.beeswarm(shap_values, max_display=20)
    st.pyplot(beeswarm)                 
                     
    st.button("Recommencer")




