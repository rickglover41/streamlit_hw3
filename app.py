#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.title("Housing Prices in Iowa")

url = r'https://raw.githubusercontent.com/rickglover41/streamlit_hw3/main/iowa_mini.csv'

num_rows = st.sidebar.number_input('Select Number of Rows to Load', min_value = 50, max_value = 1500, step = 50)

section = st.sidebar.radio('Choose Application Section', ['Look at the Data', 'Make a Prediction'])

# add the decorator for the cache (cache is an intermediate Python function to learn)
@st.cache
def load_data(num):
	df = pd.read_csv(url, nrows = num)
	return df

@st.cache
def create_grouping(x, y):
	grouping = df.groupby(x)[y].mean()
	return grouping

def load_model():
	with open('pipe_ohe.pkl', 'rb') as pickled_mod:
		model = pickle.load(pickled_mod)
	return model

df = load_data(num_rows)

if section == 'Look at the Data':
	x_axis = st.sidebar.selectbox('Choose column for X-axis', df.select_dtypes(include = np.object).columns.tolist())
	y_axis = st.sidebar.selectbox('Choose column for y-axis', ['SalePrice'])
	chart_type = st.sidebar.selectbox('Choose your chart type', ['line', 'bar', 'area'])
	
	if chart_type == 'line':
		grouping = create_grouping(x_axis, y_axis)
		st.line_chart(grouping)
	elif chart_type == 'bar':
		grouping = create_grouping(x_axis, y_axis)
		st.bar_chart(grouping)
	elif chart_type == 'area':
		fig = px.strip(df[[x_axis, y_axis]], x=x_axis, y=y_axis)
		st.plotly_chart(fig)
	
	st.write(df)

else:
	st.text('Choose Options to the Side to Get a Predicted Sales Price')
	model = load_model()
	overall_qual = st.sidebar.slider('Overall quality of the house?', min_value = 1, max_value = 10, value = 5, step = 1)
	gr_liv_area = st.sidebar.number_input('Square footage of the house?', min_value = 100, max_value = 10000, value = 2000, step = 100)
	lot_area = st.sidebar.number_input('Size of the lot?', min_value = 1000, max_value = 500000, value = 10000, step = 1000)
	year_blt = st.sidebar.number_input('What year was the house built?', min_value = 1800, max_value = 2010, value = 2000, step = 1)
	garage_blt = st.sidebar.number_input('What year was the garage built? (0 if no garage)', min_value = 0, max_value = 2010, value = 2000, step = 1)
	garage_cars = st.sidebar.slider('How many cars can the garage hold? (0 if not garage)', min_value = 0, max_value = 4, value = 0, step = 1)
	garage_type = st.sidebar.selectbox('What type of garage? (nan if no garage)', df['GarageType'].unique().tolist())
	
	sample = {
	'OverallQual': overall_qual,
	'GrLivArea': gr_liv_area,
	'1stFlrSF': gr_liv_area/2,
	'2ndFlrSF': gr_liv_area/2,
	'LotArea': lot_area,
	'YearBuilt': year_blt,
	'GarageYrBlt': garage_blt,
	'GarageCars': garage_cars,
	'GarageType': garage_type
	}
	
	sample = pd.DataFrame(sample, index = [0])
	prediction = model.predict(sample)[0]
	
	st.title(f"Predicted Sale Price: ${int(prediction)}")

