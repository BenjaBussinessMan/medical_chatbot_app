# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 00:33:13 2021

@author: info
"""

import streamlit as st
import pandas as pd
import joblib
from transformers import BertTokenizer, BertForSequenceClassification

_model = pd.read_pickle("df_label.pkl")


model_filename = 'model\df_label.pkl'



def load_model_and_predict(harness_size):
    '''
    '''

    # Cargamos el modelo desde el archivo
    loaded_model = joblib.load(model_filename)

    print("Hemos cargado el modelo con los siguientes parámetros:")
    print(loaded_model.params)

    # Preparar los datos de entrada para el modelo
    inputs = { 'harness_size' : [harness_size] }

    # Usamos el modelo para hacer predicciones
    predicted_boot_size = loaded_model.predict(inputs)[0] #devuelve una serie, usamos el [0] para que me de el valor de la celda con el resultado

    return predicted_boot_size

def check_size_of_boots(selected_harness_size, selected_boot_size):
    '''
    Calcula si el cliente ha elegido un par de botas para perros que 
    son de una talla razonable.
    
    Esto funciona estimando el tamaño real de la bota del perro 
    del perro a partir de su talla de arnés.

    Esto devuelve un mensaje para el cliente que debe mostrarse antes de que
    de completar el pago. 

    selected_harness_size: El tamaño del arnés que el cliente quiere comprar
    selected_boot_size: El tamaño de las botas que el cliente quiere comprar
    '''

    # Estimar la talla de la bota del perro del cliente
    estimated_boot_size = load_model_and_predict(selected_harness_size)

    # Redondea al número entero más cercano porque no vendemos tallas parciales
    estimated_boot_size = int(round(estimated_boot_size))

    # Verificar si el tamaño de la bota es el apropiado
    if int(selected_boot_size) == estimated_boot_size:
        # Las botas probablemente están OK
        return st.success("¡Gran elección! Creemos que estas botas se adaptarán bien a su perro.")

    if int(selected_boot_size) < estimated_boot_size:
        # Las botas seleccionadas podrían ser muy pequeñas 
        return st.error("Las botas que has seleccionado podrían ser DEMASIADO PEQUEÑAS para un perro tan "\
               f"grande como el suyo. Recomendamos unas botas de tamaño {estimated_boot_size}.")

    if int(selected_boot_size) > estimated_boot_size:
        # Las botas selecionadas podrian ser muy grandes 
        return st.error("Las botas que has seleccionado podrían ser DEMASIADO GRANDES para un perro tan "\
               f"pequeño como el suyo. Recomendamos unas botas de tamaño {estimated_boot_size}.")

 
st.title('Compra de Arneses y Botas para perros')
st.header("Tienda RED")
st.subheader("Ingrese los datos de su perro")


with st.form(key='diabetes-pred-form'):
    col1, col2 = st.columns(2)
    
    arnes = col1.slider(label='Tamaño del arnés:', min_value=0, max_value=100)
    botas = col2.text_input(label='Tamaño de la Bota:')
    submit = st.form_submit_button(label='Check')
    
    if submit:
        check_size_of_boots(selected_harness_size=(arnes), selected_boot_size=(botas))        
