#!/usr/bin/env python
# coding: utf-8




from flask import Flask, jsonify, request
import pandas as pd
from flask import  render_template
import numpy as np
from sklearn import linear_model
import joblib
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


def first_letter(col):
    if (col[0] == 'E' or col[0] == 'V'):
        return '7777'
    else:
        return col
def cat_col(col):
    if (col >= 390) & (col <= 459) | (col == 785):
        return 'circulatory'
    elif (col >= 460) & (col <= 519) | (col == 786):
        return 'respiratory'
    elif (col >= 520) & (col <= 579) | (col == 787):
        return 'digestive'
    elif (col >= 250.00) & (col <= 250.99):
        return 'diabetes'
    elif (col >= 800) & (col <= 999):
        return 'injury'
    elif (col >= 710) & (col <= 739): 
        return 'musculoskeletal'
    elif (col >= 580) & (col <= 629) | (col == 788):
        return 'genitourinary'
    elif ((col >= 290) & (col <= 319) | (col == 7777) | 
          (col >= 280) & (col <= 289) | 
          (col >= 320) & (col <= 359) |
          (col >= 630) & (col <= 679) |
          (col >= 360) & (col <= 389) |
          (col >= 740) & (col <= 759)):
        return 'other'
    else:
        return 'neoplasms' 
    
def binary_lab_procedures(col):
    if (int(col) >= 1) & (int(col) <= 10):
        return '[1-10]'
    if (int(col) >= 11) & (int(col) <= 20):
        return '[11-20]'
    if (int(col) >= 21) & (int(col) <= 30):
        return '[21-30]'
    if (int(col) >= 31) & (int(col) <= 40):
        return '[31-40]'
    if (int(col) >= 41) & (int(col) <= 50):
        return '[41-50]'
    if (int(col) >= 51) & (int(col) <= 60):
        return '[51-60]'
    if (int(col) >= 61) & (int(col) <= 70):
        return '[61-70]'
    if (int(col) >= 71) & (int(col) <= 80):
        return '[71-80]'
    if (int(col) >= 81) & (int(col) <= 90):
        return '[81-90]'
    if (int(col) >= 91) & (int(col) <= 100):
        return '[91-100]'
    if (int(col) >= 101) & (int(col) <= 110):
        return '[101-110]'
    if (int(col) >= 111) & (int(col) <= 120):
        return '[111-120]'
    else:
        return '[121-132]' 


def preprocess(data_point):
#     data.reset_index(inplace=True)
    columns_to_be_considered=['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight',
       'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
       'time_in_hospital', 'payer_code', 'medical_specialty',
       'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1',
       'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed']
    data = pd.DataFrame([data_point],columns = columns_to_be_considered)
    data.rename({'time_in_hospital':'no_of_days_admitted'},inplace=True)
    data['num_visits'] = data.groupby('patient_nbr')['patient_nbr'].transform('count')
    data.sort_values(by = 'patient_nbr', ascending = True,inplace=True)
    data.sort_values(['patient_nbr', 'encounter_id'],inplace=True)
    data.drop_duplicates(['patient_nbr'],inplace=True)
    data=data[((data.discharge_disposition_id != 11) & 
                                          (data.discharge_disposition_id != 13) &
                                          (data.discharge_disposition_id != 14) & 
                                          (data.discharge_disposition_id != 19) & 
                                          (data.discharge_disposition_id != 20) & 
                                          (data.discharge_disposition_id != 21))]
    data.groupby('discharge_disposition_id').size()
    data = data[((data.race != '?'))]
    data.replace(to_replace='?', value=np.nan, inplace=True)
    data = data.drop(['weight', 'medical_specialty', 'payer_code'], axis = 1)
    data = data[((data.diag_1 != '?') &
                                (data.diag_2 != '?') &
                                (data.diag_3 != '?'))]
    d1 = pd.DataFrame(data.diag_1.apply(lambda col: first_letter(str(col))), dtype = 'float')
    d2 = pd.DataFrame(data.diag_2.apply(lambda col: first_letter(str(col))), dtype = 'float')
    d3 = pd.DataFrame(data.diag_3.apply(lambda col: first_letter(str(col))), dtype = 'float')
    data = pd.concat([data, d1, d2, d3], axis = 1)
    data.columns.values[47:50] = ('Diag1', 'Diag2', 'Diag3')
    data = data.drop(['diag_1', 'diag_2', 'diag_3'], axis = 1)
    data['first_diag'] = data.Diag1.apply(lambda col: cat_col(col))
    data['second_diag'] = data.Diag2.apply(lambda col: cat_col(col))
    data['third_diag'] = data.Diag3.apply(lambda col: cat_col(col))
    data.rename(columns={'glyburide-metformin': 'glyburide_metformin',
                       'glipizide-metformin': 'glipizide_metformin',
                       'glimepiride-pioglitazone': 'glimepiride_pioglitazone',
                       'metformin-rosiglitazone': 'metformin_rosiglitazone',
                       'metformin-pioglitazone': 'metformin_pioglitazone', }, inplace=True)
    data = data.drop(['encounter_id', 'patient_nbr', 'Diag1', 'Diag2', 'Diag3'], axis = 1)
    data = data.replace('?', np.NaN)
    data.loc[(data.gender == 'Unknown/Invalid'),'gender']='Female'    
    data['HbA1c'] = np.where(data['A1Cresult'] == 'None', 0, 1)
    data['num_lab_procedure_ranges'] = data['num_lab_procedures'].apply(lambda x: binary_lab_procedures(x))
    data=data.drop(['num_lab_procedures'], axis = 1)
    columns = data[['admission_type_id', 'discharge_disposition_id', 'admission_source_id']] 
    data[['admission_type_id', 'discharge_disposition_id', 'admission_source_id']] = columns.astype(object)
    data_encoded=data.apply(LabelEncoder().fit_transform)
    return data_encoded


# In[3]:


@app.route('/')
def hello_world():
    return 'Hello World!'


# In[4]:


@app.route('/index')
def index():
    return flask.render_template('index.html')


# In[6]:


@app.route('/fun1', methods=['POST'])
def fun1():
    best_model = joblib.load('stacking_classifier_model_final_last.pkl')
    data=[x for x in request.form.values()]
    data_encoded=preprocess(data)
#     print(data_encoded)
    label = best_model.predict(data_encoded)
#     print('The label is : ', label)
    if label[0]:
        prediction = "not-readmitted"
    else:
        prediction = "yes-readmitted"

    return render_template('index.html',prediction_text="Prediction is {}".format(prediction))


# In[ ]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




