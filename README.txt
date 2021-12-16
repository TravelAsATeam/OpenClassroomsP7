Projet OpenClassrooms : Implémenter un modèle de scoring

Le projet consiste en la recherche d'un modèle de prédiction de défaut de paiement de crédit. 

Il inclut :

-une API de prédiction déployée sur le cloud sur la plateforme https://www.pythonanywhere.com à l'adresse 
http://travelasateam.pythonanywhere.com/predict/

- un dashboard permettant de visualiser des données, de faire appel à l'API de prédiction et de fournir des informations sur l'interprétabilité locale et globale du modèle, il est dispoible à l'adresse
https://share.streamlit.io/travelasateam/openclassroomsp7/main/dashboard.py

- un notebook détaillant la recherche du modèle

- deux jeux de données provenant de l'échantillon principal mis à l'échelle ou non pour visualisation

- le code du dashboard et de l'API de prédiction via flask

- le kernel kaggle récupéré sur la platefore, création publique d'un autre auteur et utilisé pour le traitement initial des données avec quelques modifications pour la sauvegarde des données ainsi traitées

- un fichier requirements pour l'API et le dashboard

- un enregistrement du dernier modèle en format pickle

- un enregistrement de l'explainer shap du dernier modèle en format pickle

- la note méthodologique