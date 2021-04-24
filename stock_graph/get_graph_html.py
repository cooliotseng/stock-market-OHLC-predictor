import pandas as pd
import plotly.offline as po
import plotly.graph_objs as go
import pandas_datareader.data as web
import datetime
def get_graph(search,start):
 try:
      sid = search
      sd = start
      ed = datetime.datetime.now()
      df = web.DataReader(sid, 'yahoo', sd, ed)
      df.columns = ['high', 'low', 'open','close','volume','adj close']
      SMA5  = df['close'].rolling(5).mean()
      SMA10 = df['close'].rolling(10).mean()
      SMA20 = df['close'].rolling(20).mean()
      SMA60 = df['close'].rolling(60).mean()
      trace = go.Candlestick(x=df.index,open=df['open'],high=df['high'],low=df['low'],close=df['close'],name = 'K')
      s5 = go.Scatter(x = SMA5.index,y = SMA5.values,name = '5MA')
      s10 = go.Scatter(x = SMA10.index,y = SMA10.values,name = '10MA')
      s20 = go.Scatter(x = SMA20.index,y = SMA20.values,name = '20MA')
      s60 = go.Scatter(x = SMA60.index,y = SMA60.values,name = '60MA')
      data = [trace,s5,s10,s20,s60]
      layout = {'title': sid}
      fig = dict(data=data, layout=layout)
      po.plot(fig, filename='templates/stock.html',auto_open=False)
      return True
 except:
    return False
