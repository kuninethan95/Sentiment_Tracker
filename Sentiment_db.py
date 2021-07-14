
from pandas.io.parsers import read_csv
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import SessionState
import joblib

st.set_page_config(layout="wide")


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
    fig.update_layout(width=1200,height=400)
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
    return df#.iloc[:,[0,1,2,3,-1]].sort_index()

df_day = show_day(df, day)
# st.write(df_day.iloc[:,[0,1,2,3,-1]].sort_index())

# Reset index before coloring, remove cols
def reset_index_version_remove_cols(df):
    df = df.reset_index()
    df.drop(columns=['description', 'content', 'results', 'hour', 'day', 'cont_joined', 'neg', 'neu',
                    'pos', 'blob', 'comp_abs', 'blob_abs', 'c_b'], inplace=True)
    df.rename(columns={'source2': 'source'}, inplace=True)
    return df

df_no_index = reset_index_version_remove_cols(df_day)
# st.write(df_no_index)

# Apply coloring to DF

num = df_no_index.shape[1]
def highlight(s):

    if s['comp'] > 0.90:
        return ['background-color: green']*num
    elif s['comp'] < -0.80:
        return ['background-color: red']*num
    else:
        return ['background-color: white']*num

df_colored_day = df_no_index.style.apply(highlight, axis=1)

def paginated_df(data):
    N = 10

    # A variable to keep track of which product we are currently displaying
    session_state = SessionState.get(page_number = 0)
    df_day.rename(columns={'source2':'Source', 'title':'Headline'},inplace=True)
    data = data
    last_page = len(data) // N

    # Add a next button and a previous button

    prev, _ ,next = st.beta_columns([1, 6, 1])

    if next.button("Next"):

        if session_state.page_number + 1 > last_page:
            session_state.page_number = 0
        else:
            session_state.page_number += 1

    if prev.button("Previous"):

        if session_state.page_number - 1 < 0:
            session_state.page_number = last_page
        else:
            session_state.page_number -= 1

    # Get start and end indices of the next page of the dataframe
    start_idx = session_state.page_number * N 
    end_idx = (1 + session_state.page_number) * N

    # Index into the sub dataframe
    sub_df = data.iloc[start_idx:end_idx, [0,4,-1]]
    return sub_df
#st.write(sub_df)

# st.write(paginated_df(df_day))


st.table(paginated_df(df_day))

# st.dataframe(df_colored_day(use_column_width=True))
# st.dataframe(df_colored_day)

def sentiment_hour(df, day):
    df_day = df[df['day']==day]
    labels = ['Positive','Neutral','Negative']
    values = list(df_day['Sentiment'].value_counts().values)
    colors = ['Green', 'yellow', 'Red']
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    return fig


pie_chart = sentiment_hour(df, day)

# col2.header("Sentiment Distribution")
# col2.plotly_chart(pie_chart)

st.plotly_chart(pie_chart)
st.write(company)
st.write(week)

def find_file(company):
    if company == "Apple":
        if week == "June 14th - June 19th":
            file = joblib.load('data/etl_df/df_etl_apple1.csv')
    if week == "June 21st - June 25th":
            file = joblib.load('data/etl_df/df_etl_apple1.csv')

    if company == "Amazon":
        if week == "June 14th - June 19th":
            file = joblib.load('data/etl_df/df_etl_apple1.csv')
        if week == "June 21st - June 25th":
            file = joblib.load('data/etl_df/df_etl_apple1.csv')

    if company == "Microsoft":
        if week == "June 14th - June 19th":
            file = joblib.load('data/etl_df/df_etl_apple1.csv')
        if week == "June 21st - June 25th":
            file = joblib.load('data/etl_df/df_etl_apple1.csv')

    if company == "Netflix":
        if week == "June 14th - June 19th":
            file = joblib.load('data/etl_df/df_etl_apple1.csv')
        if week == "June 21st - June 25th":
            file = joblib.load('data/etl_df/df_etl_apple1.csv')

    return file


def feat_imp_fig(pkl_file, company, week):
    

    # Load in pickle file
    comp_be = joblib.load(pkl_file)

    # Create vector + clf
    vect = comp_be.named_steps['text_pipe']
    clf = comp_be.named_steps['clf']

    # Feature Importance Series
    importance = pd.Series(clf.feature_importances_,
                          index=vect.get_feature_names())

    importance=importance.sort_values(ascending=False).to_frame()
    importance = importance.nlargest(30, columns=0)
    importance=importance.sort_values(by=0, ascending=True)

    # Remove company name
    importance.drop(company, inplace=True)

    # Rename column
    importance.rename(columns={0:'Sentiment Importance'}, inplace=True)

    # Create Plotly fig
    fig = px.bar(importance, y=importance.index, x='Sentiment Importance', orientation='h')
    layout1 = go.Layout(
    title={
        'text': f"Week of {week}: Most Impactful Words for Driving Sentiment for {company.capitalize()}",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        },
    yaxis=dict(
               tickvals=importance.index,
        title='Commmon Words',
        ))
    fig.update_layout(layout1)
    return fig

fig = feat_imp_fig('grid_estimator_models/apple1_bp.pkl', company, week)
st.plotly_chart(fig)