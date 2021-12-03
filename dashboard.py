#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import seaborn as sns
import os

# Téléchargement des données
default_dir = os.getcwd()
data = pd.read_csv(os.path.join(default_dir,'data_sampled.csv'))

# Choix du mode de fonctionnement
mode_predict = False
if st.button("Faire une prediction"):
    mode_predict = True

# Mode affichage de graphique
if mode_predict == False :
    features = st.multiselect("Choisissez deux variables", list(data.columns))
    if len(features) != 2 :
        st.error("Sélectionnez deux variables")
    else :
        st.write("## Graphique bi-varié")
        # Graphique
        chart = sns.jointplot(x=data[features[0]], y=data[features[1]], height=10)
        # Regression linéaire
        sns.regplot(x=data[features[0]], y=data[features[1]], scatter=False, ax=chart.ax_joint)
        
# Mode prédiction
if mode_predict == True :
    st.error("Non implementé")
 


# In[5]:


1 !=2


# In[ ]:




