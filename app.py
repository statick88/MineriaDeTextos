import pandas as pd  # Corrige la importación de pandas
import streamlit as st
from gensim.models.phrases import Phrases, Phraser  # Asegura que Phraser esté correctamente importado
from nltk.tokenize import word_tokenize

st.title("Detección de Colocaciones en Texto")

# Ingresar texto
text = st.text_area("Ingrese un texto:")

# Cargar el modelo de bigramas
to_collocations = Phraser.load('bigram_model')

# Tokenizar el texto ingresado
tokens = word_tokenize(text.lower())

# Aplicar el modelo de bigramas
colocaciones = to_collocations[tokens]

# Mostrar las colocaciones detectadas
st.write("Colocaciones Detectadas:")
st.write(colocaciones)

# Mostrar las colocaciones en una tabla
st.write(pd.Series(colocaciones).value_counts())
