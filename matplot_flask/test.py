from flask import Flask, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




app = Flask(__name__)

import pandas as pd
import mplfinance as mpl
stock_list = ['BAJFINANCE','BAJFINANCE','BAJFINANCE']

@app.route('/')
def index():  

    return render_template('index.html', stocks=stock_list)
 

if __name__ == '__main__':
   app.run(debug = True)
