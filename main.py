import streamlit as st
import pandas as pd
import numpy as np
from os import path
import pickle #used to read .pkl file


# st.title('Iris dataset')
# df_iris = pd.read_csv(path.join("Data","iris.csv"))
# #filepath = C:\Users\soora\Downloads\personal-projects\Streamlit_demo\data\iris.csv
#
# st.write(df_iris)
#
# st.scatter_chart(df_iris[['sepal_length','sepal_width']])

st.title("Iris Species predictor")
petal_length = st.number_input("Choose your petal length",
                               placeholder='please enter a valid number between 1 and 6.9',
                               min_value=1.0,
                               max_value=6.9,
                               value=None)
petal_width = st.number_input("choose the petal_width",
                         placeholder='please enter a valid number between 0.1 and 2.5',
                         min_value=0.1,
                         max_value=2.5,
                         value=None)
sepal_length = st.number_input(" choose your sepal length",
                               placeholder='please enter a valid number between 4.3 and 6.9',
                               min_value=4.3,
                               max_value=7.9,
                               value=None)
sepal_width = st.number_input("please enter your Sepal width",
                              placeholder='please enter a valid number between 2.0 and 4.4' ,
                              min_value=2.0,
                              max_value=4.4,
                              value=None)
#prepare a dataframe for the prediction
df_user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length,petal_width]],
    columns = ['sepal_length','sepal_width','petal_length','petal_width'])


st.write(df_user_input)
#using the pkl file creating an ml model named iris_predictor
model_path = path.join("model","iris_classifier.pkl")
with open(model_path, 'rb') as file:
    iris_predictor =pickle.load(file)

#decode
#'setosa'#'versicolor'#'virginica'
dict_species = {0:'setosa',1:'versicolor',2:'virginica'}

if st.button("Predict species"):
     if ((petal_length==None) or (petal_width==None) or(sepal_length==None) or (sepal_width==None)):
         st.write("please fill all the values") #will not be excecuted if any of thr values not entered properly
     else:
         #prediction can be done here
         predicted_species=iris_predictor.predict(df_user_input)
         st.write("species is :",dict_species[predicted_species[0]])