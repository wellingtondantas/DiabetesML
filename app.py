#Libries
import pandas as pd 
import numpy as np
import streamlit as st

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#title
st.markdown("<h1 style='color:#87CEEB;'>Modelo de Predição de Diabetes</h1>", unsafe_allow_html=True)
#sub-title
st.subheader("""
App que utiliza inteligência articial para prever possível caso de diabetes em pessoas. \n\n
""")


# ----------------------START MODEL ----------------------------

#load data csv
df = pd.read_csv('https://raw.githubusercontent.com/wellingtondantas/Data-Science/master/Datasets/diabetes.csv')

X = df.drop(['Outcome'], axis=1)
Y = df['Outcome']

#Data 80:20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#Train model
model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
model.fit(X_train, y_train)

#Predict
classes = model.predict(X_test)
st.subheader('Acurácia do modelo atualmente')
st.write(accuracy_score(y_test,classes))




# ----------------------THE END MODEL ----------------------------

#information
st.markdown("<h2 style='color:#F00;'><br />Digite as informações do menu ao lado:</h2>", unsafe_allow_html=True)


#data input
input_user = st.sidebar.text_input('Digite seu Nome')
input_Pregnancies = st.sidebar.slider('Número de Gravidez', 0, 17, 0)
input_Glucose = st.sidebar.slider('Nível de glicose', 0, 199, 100)
input_BloodPressure = st.sidebar.slider('Nível de Pressão Sanguínea (mm Hg)', 0, 122, 70)
input_SkinThickness = st.sidebar.slider('Espessura da Pele (mm)', 0, 99, 20)
input_Insulin = st.sidebar.slider('Nível de Insulina (mu U/ml)', 0, 846, 50)
input_BMI = st.sidebar.slider('Índice de Massa Corporal', 0.0, 67.10, 15.0)
input_DiabetesPedigreeFunction = st.sidebar.slider('Você tem Historio de Diabetes na Família', 0.0, 2.5, 1.0)
input_Age = st.sidebar.text_input('Digite sua Idade', 0)


#Join inputs
user_data = {'Número de Gravidez':input_Pregnancies,
             'Nível de glicose':input_Glucose,
             'Nível de Pressão Sanguínea (mm Hg)':input_BloodPressure,
             'Espessura da Pele (mm)':input_SkinThickness,
             'Nível de Insulina (mu U/ml)':input_Insulin,
             'Índice de Massa Corporal':input_BMI, 
             'Historio de Diabetes na Família':input_DiabetesPedigreeFunction, 
             'Digite sua Idade':input_Age}

#Creat feature test
simple_feature = pd.DataFrame(user_data, index=[0])

#Show features pacient
st.write('Dados do Paciente:', input_user)
st.write(user_data)


# ------------------ PREDICT PACIENT -------------------
predic_pacient = model.predict(simple_feature)

st.subheader('Predição Status:')

if input_user == '':
    st.write('Sem dados')
elif predic_pacient == 0 :
    st.write('Paciente sem Risco de Diabetes')
elif predic_pacient == 1 :
    st.write('Paciente com Risco de Diabetes')
else:
    st.write('Sem dados')
