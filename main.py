# SVM has high accuracy
# (7th,8th,9th) questions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import joblib
df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t')
x=df['Review'].values
y=df['Liked'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.24,random_state=0)
# count vectorizer
cv=CountVectorizer()
vect_x_train=cv.fit_transform(x_train).toarray()
vect_x_test=cv.transform(x_test).toarray()
model=SVC().fit(vect_x_train,y_train)
y_pred=model.predict(vect_x_test)
accu=accuracy_score(y_pred,y_test)
model1=make_pipeline(CountVectorizer(),SVC()).fit(x_train,y_train)
y_pred1=model1.predict(x_test)
final=joblib.dump(model1,'project.pkl')
final_model=joblib.load('project.pkl')
print(final_model.predict([' not nice ']))

















