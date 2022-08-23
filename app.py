import string

import joblib
import streamlit as st

model=joblib.load('project.pkl')
#cv=joblib.load(open('cv.pkl','rb'))
st.title('Sentimental Analysis ')
st.subheader('---BY Ajmeera Kiranchanra---')
sen=st.text_area("Write a review")

if not sen:
    st.warning("Please fill out required fields")

if st.button('predict'):
    prediction=model.predict([sen])
    #print(prediction)
    if prediction==[0]:
        original_title = '<p style="font-family:Courier; color:red; font-size: 20px;">The review is Negative.</p>'

        st.markdown(original_title, unsafe_allow_html=True)

        #st.success(original_title)
    else:

        st.success('The review is Positive')


















