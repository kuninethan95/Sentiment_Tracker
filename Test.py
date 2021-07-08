
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# Set title
st.title("Sentiment Tracker")

# Add description
st.write("""
## Explore the sentiment of different companies
Choose between Apple, Facebook, Microsoft, Netlfix, and Google""")

# Select company to analyze
company = st.selectbox(
    'Select Company',
    ('Apple', 'Facebook',  'Microsoft', 'Google', 'Netflix'))

# Choose a week to get data from
week = st.selectbox('Select Week', ('June 14th - June 19th', 'June 21st - June 25th'))

# Get dataset for corresponding company and week
def get_dataset(company, week):
    
    if company == "Apple":
        if week == "June 14th - June 19th":
            df = pd.read_csv('data/etl_df/df_etl_apple1.csv')
        if week == "June 21st - June 25th":
            df = pd.read_csv('data/etl_df/df_etl_apple2.csv')
    
    if company == "Amazon":
        if week == "June 14th - June 19th":
            df = pd.read_csv('data/etl_df/df_etl_am1.csv')
        if week == "June 21st - June 25th":
            df = pd.read_csv('data/etl_df/df_etl_am2.csv')
    
    if company == "Microsoft":
        if week == "June 14th - June 19th":
            df = pd.read_csv('data/etl_df/df_etl_msft1.csv')
        if week == "June 21st - June 25th":
            df = pd.read_csv('data/etl_df/df_etl_msft2.csv')
    
    if company == "Netflix":
        if week == "June 14th - June 19th":
            df = pd.read_csv('data/etl_df/df_etl_nflx1.csv')
        if week == "June 21st - June 25th":
            df = pd.read_csv('data/etl_df/df_etl_nflx2.csv')

    return df

# DF = specified dataframe and week
df = get_dataset(company, week)

# Set index and hour/day
def set_idx(df):
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df.set_index('publishedAt', inplace=True)
    df.index = df.index.tz_convert('US/Eastern')
    df.sort_index(inplace=True)
    df['hour'] = df.index.hour
    df['day']=df.index.day
    return df

# Function to set index
df_idx = set_idx(df)

# Resample for visualization
def resample(df):
  df = df.resample('h').mean()
  df.dropna(inplace=True)
  df=df[~df['hour'].isin([16])].copy()
  df['results'] = df['results'].astype('int')
  return df

df_resamp = resample(df_idx)


# Sentiment tracker visualization
def sent_vol_fig(df_, company):

    df = resample(df_)
    df['roll']=df['Sentiment'].rolling(window=5).mean().dropna()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
      go.Scatter(mode="lines+markers",opacity=1.0,x=df.index, y=df['Sentiment'], name="Sentiment"),
      secondary_y=False,
    )

    fig.add_trace(
      go.Scatter(mode="lines+markers",opacity=1.0,x=df.index, y=df['roll'], name="*Rolling* Sentiment"),
      secondary_y=False,
    )

    fig.add_trace(
      go.Bar(opacity=0.8, x=df.index, y=df['results'], name="Volume"),
      secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
      title_text=f"{company} Sentiment Graph"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date - Time")

    # Set y-axes titles
    
    fig.update_yaxes(title_text="<b>Sentiment</b>", secondary_y=False)
    if company!='Netflix':
        fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=True, range=[40,200])
        fig.update_xaxes(rangebreaks=[dict(bounds=[16, 9], pattern="hour")])
    else:
        fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=True, range=[20,150])
        fig.update_xaxes(rangebreaks=[dict(bounds=[16, 9], pattern="hour")])
    
    fig.update_xaxes(rangeslider_visible=True)
    return fig

plotly_fig = sent_vol_fig(df_resamp, company)

st.plotly_chart(plotly_fig)

st.write('Please choose a weekday to further inspect')

if week == 'June 14th - June 19th':
    date = st.date_input('Date input', value=datetime.datetime(2021, 6,14), max_value=datetime.datetime(2021, 6, 18), min_value=datetime.datetime(2021, 6, 14))
elif week == 'June 21st - June 25th':
    date = st.date_input('Date input', value=datetime.datetime(2021, 6,21), max_value=datetime.datetime(2021, 6, 25), min_value=datetime.datetime(2021, 6, 21))

# date=date
date_str = str(date)

# Make acceptable and unacceptable dates
def aceptable_date(date_str):
    if date_str == '2021-06-19':
        return ('Please choose weekday')
    else:
        return ''

st.write(aceptable_date(date_str))

# Set day
day = date.day

# Show dataframe for specific day
def show_day(df, day):
    df = df[df['day']==day]
    return df.iloc[:,[0,1,2,3,-1]].sort_index()

# Apply coloring to DF
df_day = show_day(df, day)

st.dataframe(df_day)

def sentiment_hour(df, day):
    df_day = df[df['day']==day]
    labels = ['Negative','Neutral','Positive']
    values = list(df_day['Sentiment'].value_counts().values)
    colors = ['Green', 'yellow', 'Red']
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    return fig

pie_chart = sentiment_hour(df, day)
st.plotly_chart(pie_chart)


