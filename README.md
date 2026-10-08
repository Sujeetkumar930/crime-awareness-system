\# Crime Awareness System



A machine learning based Crime Awareness System developed using Python, Pandas, Scikit-learn and Streamlit.



\## Project Overview



The Crime Awareness System allows users to select a State, District, City, Crime Type and Year. The system uses a trained machine learning model to predict the crime count and also displays historical crime information and state-level information.



\## Features



\- State selection

\- District selection

\- City selection

\- Crime type selection

\- Year selection

\- Crime count prediction

\- Historical crime count lookup

\- State information

\- Simple crime awareness chatbot



\## Technologies Used



\- Python

\- Pandas

\- Scikit-learn

\- Streamlit

\- Joblib

\- Machine Learning



\## Machine Learning



The project uses a Random Forest Regression model for predicting crime counts.



The categorical values such as State, District, City and Crime Type are converted into numerical values using Label Encoding.



\## Dataset



The project uses multiple crime datasets. The processed datasets contain:



\- State

\- District

\- City

\- Year

\- Crime Type

\- Crime Count



The individual datasets are combined to create the master dataset used for machine learning.



\## Application



The application is developed using Streamlit.



To run the application:



```bash

streamlit run app.py

