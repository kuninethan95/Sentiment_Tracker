
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Sentiment Tracker")

st.write("""
## Explore the sentiment of different companies
Choose between Apple, Facebook, Microsoft, Netlfix, and Google""")

company = st.selectbox(
    'Select Company',
    ('Apple', 'Facebook',  'Microsoft', 'Google', 'Netflix'))

week = st.selectbox('Select Week', ('June 14th - June 19th', 'June 21st - June 25th'))

def get_dataset(company, week):
    
    if company == "Apple":
        if week == "June 14th - June 19th":
            df = pd.read_csv('good_apple.csv')
        if week == "June 21st - June 25th":
            df = pd.read_csv('data/news_articles/hourly_apple_articles_621_625')
    
    if company == "Amazon":
        if week == "June 14th - June 19th":
            df = pd.read_csv('data/news_articles/hourly_am_articles_614_618')
        if week == "June 21st - June 25th":
            df = pd.read_csv('data/news_articles/hourly_am_articles_621_625')
    
    if company == "Microsoft":
        if week == "June 14th - June 19th":
            df = pd.read_csv('data/news_articles/hourly_msft_articles_614_618')
        if week == "June 21st - June 25th":
            df = pd.read_csv('data/news_articles/hourly_msft_articles_621_625')
    
    if company == "Netflix":
        if week == "June 14th - June 19th":
            df = pd.read_csv('data/news_articles/hourly_nf_articles_614_618')
        if week == "June 21st - June 25th":
            df = pd.read_csv('data/news_articles/hourly_nf_articles_621_625')

    return df

df = get_dataset(company, week)

st.write(df)

def set_idx(df):
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df.set_index('publishedAt', inplace=True)
    df.index = df.index.tz_convert('US/Eastern')
    df.sort_index(inplace=True)
    df['hour'] = df.index.hour
    df['day']=df.index.day
    return df

df_idx = set_idx(df)

def resample(df):
  df = df.resample('h').mean()
  df.dropna(inplace=True)
  df=df[~df['hour'].isin([16])].copy()
  df['results'] = df['results'].astype('int')
  return df

df_resamp = resample(df_idx)
st.write(df_resamp)

def sent_vol_fig(df, company):

  df['roll']=df['rules_sent'].rolling(window=5).mean().dropna()

  fig = make_subplots(specs=[[{"secondary_y": True}]])

  # Add traces
  fig.add_trace(
      go.Scatter(mode="lines+markers",opacity=1.0,x=df.index, y=df['rules_sent'], name="Sentiment"),
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
  fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=True, range=[40,200])
  fig.update_xaxes(rangebreaks=[dict(bounds=[16, 9], pattern="hour")])

  fig.update_xaxes(rangeslider_visible=True)
  return fig

plotly_fig = sent_vol_fig(df_resamp, company)

st.plotly_chart(plotly_fig)

