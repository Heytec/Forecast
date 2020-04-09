import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
#import plotly.express as px


#The st.cache decorator indicates that Streamlit will perform internal magic so that the data will be downloaded only once and cached for future use.



st.header("Forecasting Website Traffic ")
st.write('Generate Quick and Accurate Time Series Forecasts using Facebook’s Prophet')

#upload data
#The st.cache decorator indicates that Streamlit will perform internal magic so that the data will be downloaded only once and cached for future use.
st.cache()
data = st.file_uploader("upload Timeseries datset", type=['csv', 'txt','xlsx'])
if data is not None:
   df = pd.read_csv(data)
   st.markdown('Display data')
   st.write(df.head())

#rename the columns
df.columns = ['ds','y']
st.markdown('Convert the dataframe according to Prophet  variable names')
st.write(df.head())

st.markdown('Data shape')
st.write(df.shape)

st.markdown('Inspect what the data looks like before feeding it into Prophet')
plt.figure(figsize=(10,6))
plt.plot(df.set_index('ds'))
plt.legend(['y'])
st.pyplot(plt)



st.markdown('Log transformed data')
df['y'] = np.log(df['y'])
plt.figure(figsize=(10,6))
plt.plot(df.set_index('ds'))
plt.legend(['y'])
st.pyplot(plt)


#  train the model
m=Prophet(daily_seasonality=True)
m.fit(df)


st.markdown('Predicted values')
# define prediction period 
future= m.make_future_dataframe(periods=30)

#predict 
forecast= m.predict(future)
#predicted value 
forecast.tail().T



#see the value normally 
#predicted data
st.write(np.exp(forecast[['yhat', 'yhat_lower', 'yhat_upper']].tail(12)))


# plot forecast
st.markdown("")
st.pyplot(m.plot(forecast))

#plot trends 
st.markdown("Trends")
st.pyplot(m.plot_components(forecast))


st.markdown("Cross-validation prediction performance")
#cross-validation to assess prediction performance on a horizon 
df_cv = cross_validation(m, initial='500 days', period='100 days', horizon = '300 days')
st.write(df_cv.head())



st.markdown("Performance metrics")
#permonce metrics 
df_p = performance_metrics(df_cv)
st.write(df_p.head())



st.markdown("Performance Plot")
fig = plot_cross_validation_metric(df_cv, metric='mape')
st.pyplot(fig)



