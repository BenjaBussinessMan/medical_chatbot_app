"""
Created on Thu Oct  7 00:33:13 2021

@author: info
"""


import streamlit as st
import pandas as pd
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

#input_dir='D:\\OneDrive - BGR\\Analytics\\medical_intent_detector_using_BERT\\model'
#input_dir=


loaded_model = BertForSequenceClassification.from_pretrained(current_dir)
loaded_model.eval()
loaded_tokenizer = BertTokenizer.from_pretrained(current_dir)
loaded_df_label = pd.read_pickle('df_label.pkl')


# test the model on an unseen example

def medical_symptom_detector(intent):

  pt_batch = loaded_tokenizer(
  intent,
  padding=True,
  truncation=True,
  return_tensors="pt")

  pt_outputs = loaded_model(**pt_batch)
  __, id = torch.max(pt_outputs[0], dim=1)
  prediction = loaded_df_label.iloc[[id.item()]]['intent'].item()
  doc='You may have a medical condition: '+prediction+' Would you like me to transfer your call to your doctor?'
  st.write(doc)
  return


st.title('Síntomas a Diagnóstico: Herramienta de Análisis Médico')



with st.form(key='diabetes-pred-form'):
    #col1, col2 = st.columns(2)
    

    paciente = st.text_input(label='Ingrese sus sintomas actuales:')
    submit = st.form_submit_button(label='Revisar')
    
    if submit:
        medical_symptom_detector(paciente)


