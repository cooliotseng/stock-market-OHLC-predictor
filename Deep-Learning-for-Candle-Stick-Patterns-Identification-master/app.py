# import required packages
import matplotlib.pyplot as plt
from nsepy import get_history
from mplfinance.original_flavor import candlestick_ohlc
from datetime import date
import pandas as pd
import matplotlib.dates as mpdates
import requests


def obtain_data(ticker,start,end):
# Enter the start and end dates using the method date(yyyy,m,dd)    
    stock=get_history(symbol=ticker,start=start,end=end, index=True)
    df=stock.copy()
    df=df.reset_index()
    
    df.index=df.Date
    return df



from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

# @app.route('/Deep-Learning-for-Candle-Stick-Patterns-Identification-master/app.py', methods=['post', 'get'])
# def chartTest():
#     if request.method == 'post':
#         df=obtain_data('NIFTY',date(2017,10,8),date(2018,10,8))

#         plt.style.use('dark_background')

#         # extracting Data for plotting

#         df = df[['Date', 'Open', 'High',
#                 'Low', 'Close']]

#         # convert into datetime object
#         df['Date'] = pd.to_datetime(df['Date'])

#         # apply map function
#         df['Date'] = df['Date'].map(mpdates.date2num)

#         # creating Subplots
#         fig, ax = plt.subplots(figsize=(10,10))

#         # plotting the data
#         candlestick_ohlc(ax, df.values, width = 0.6,
#                         colorup = 'green', colordown = 'red',
#                         alpha = 0.8)

#         # allow grid
#         ax.grid(True)

#         # Setting labels
#         ax.set_xlabel('Date')
#         ax.set_ylabel('Price')

#         # setting title
#         plt.title('Prices For the Period 01-07-2020 to 15-07-2020')

#         # Formatting Date
#         date_format = mpdates.DateFormatter('%d-%m-%Y')
#         ax.xaxis.set_major_formatter(date_format)
#         fig.autofmt_xdate()

#         fig.tight_layout()

#         # show the plot
#         # plt.show()

#         return render_template('index.html', name = plt.show())


# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('home'))
    return render_template('index.html', error=error)



if __name__ == '__main__':
   app.run(debug = True)