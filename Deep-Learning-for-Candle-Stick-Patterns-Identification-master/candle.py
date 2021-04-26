import pandas as pd
import mplfinance as mpl

data = pd.read_csv('AMZN.csv')

data.Date = pd.to_datetime(data.Date)

data = data.set_index('Date')


mpl.plot(data['2021-02':'2021-03'], type='candle', style='yahoo',
        title='Amazon Price Chart',
        volume=True, 
        tight_layout=True,
        figsize=(10,10))