# coding: utf-8

import streamlit as st
import datetime
import matplotlib.pyplot as plt
import gc
import time
from urllib.request import urlopen,Request
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import functions


finviz_url='https://finviz.com/quote.ashx?t='
markets=['AMZN','TSLA','FB','NFLX','GOOGL','AAPL','MSFT']

news_tables = {}

for market in markets:
    url = finviz_url + market

    req = Request(url=url, headers={'user-agent': 'my-app'})
    reponse = urlopen(req)

    html = BeautifulSoup(reponse, 'html')
    news_table = html.find(id="news-table")
    news_tables[market] = news_table


parsed_data=[]
for market,news_table in news_tables.items():

  for row in news_table.findAll('tr'):
    title=row.a.text
    date_data=row.td.text.split(' ')

    if len(date_data)==1:
      time=date_data[0]

    else:
      time=date_data[1]
      date=date_data[0]
    parsed_data.append([market,date,time,title])


df = pd.DataFrame(parsed_data, columns=['market', 'date', 'time', 'title'])
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()
f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)
df['date'] = pd.to_datetime(df.date).dt.date
mean_df = df.groupby(['market', 'date']).mean()
mean_df = df.groupby(['market', 'date']).mean().unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()

import functions


# img vars
img = plt.imread("img/pexels-mamunurpics-3930012.jpg")
x_width_image = 4892
horizon_height = 1480

################################################################################
# App layout
################################################################################

# Sidebar
st.sidebar.markdown("Look up ticker symbols [here](https://finance.yahoo.com/lookup)")
ticker_symbol = st.sidebar.text_input('Stock ticker symbol', value='AAPL')
start_date = st.sidebar.date_input("Start day", datetime.date(2010, 8, 14))
end_date = st.sidebar.date_input("End day", datetime.date(2020, 8, 14))

# Main Window

st.title("Stock Price Chart")





# Function calls to get data and make image
df_ticker = functions.get_stock_data(ticker_symbol, start_date, end_date)
stock_prices = functions.prepare_data(df_ticker['Close'])
fig = functions.make_picture(stock_prices, img=img, x_width_image=x_width_image, horizon_height=horizon_height)
st.pyplot(fig=fig, bbox_inches='tight')
plt.close(fig)



fig, ax = plt.subplots()
mean_df.plot(kind='bar',figsize=(20,8),width=1.5,ax=ax)
ax.set_title("Sentiment Analysis of Trending Stocks")
st.pyplot(fig)




gc.collect()










st.markdown("Suggestions [welcome](https://github.com/frason88). Image source: [mamunurpics](https://www.pexels.com/@mamunurpics). Inspired by [stoxart](https://www.stoxart.com).")


max_width_str = f"max-width: 1000px;"
st.markdown(f"""<style>.reportview-container .main .block-container{{ {max_width_str} }}</style>""", unsafe_allow_html=True)

