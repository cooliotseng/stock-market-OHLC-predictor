from flask import render_template, Flask, request
from get_graph_html import get_graph
app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def homepage():
    try:
        if request.method == 'POST':
            search = request.form['search']
            start = request.form['start']
            result = get_graph(search,start)
            if result == True:
                return render_template("stock.html")
            else:
                return render_template('main.html', sign='notfound')
        else:
            return render_template('main.html')
    except Exception as e:
        return render_template("main.html", sign=e)
