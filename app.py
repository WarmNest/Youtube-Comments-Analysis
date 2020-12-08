from flask import Flask, render_template, request, redirect
import pandas as pd
import searchbykey
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.pyplot as plt
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)

# main page
@app.route("/")
def firstpage():
    return render_template('index.html')

@app.route("/output_test")
def output_test():
    data=pd.read_csv("output_test.csv")
    point=0
    for i in range(0,len(data)):
        if str(data['class'].iloc[i])==str(data['label'].iloc[i]):
            point=point+1
    num=len(data)
    percent=round(point/num,3)*100
    return render_template('accuracyshow.html',datas=data,acc=percent,len=num)

# about page
@app.route("/about.html")
def aboutpage():
    return render_template('about.html')

# page for showing the result of query seach
@app.route("/showlistpage", methods=['POST'])
def showlistpage():
    if request.method == "POST":
        query = request.form['keyword']
        query_results = searchbykey.show_list(query)
        num = len(query_results['video_title'])
        return render_template('showlistpage.html', datas=query_results, len=num, keyword=query)

# page for show plot from output sentiment analysis
@app.route("/choosevideo/<string:keyword>/<int:row>", methods=['GET'])
def choosevideo(keyword, row):
    video = searchbykey.choosevideo(keyword, row)
    data = searchbykey.sentiment(video)
    # get data in form of dataframe
    count_class = data['class'].value_counts().to_frame()

    # generate plot
    pngImage = count_class.plot.pie(y='class', autopct="%.1f%%", colors=[
                                    'deepskyblue', 'aquamarine', 'salmon'], shadow=True).get_figure()

    # save plot in base64 for bringing the plot to html file without saving in png
    buf = io.BytesIO()
    pngImage.savefig(buf, format='png', transparent=True) 
    buf.seek(0)
    buffer = b''.join(buf)
    pngImageB64String = base64.b64encode(buffer).decode('utf8')
    plt.clf()
    fig = searchbykey.wordcloud(data)

    buf2 = io.BytesIO()
    fig.savefig(buf2, format='png', transparent=True)
    buf2.seek(0)
    buffer2 = b''.join(buf2)
    pngImageB64String_2 = base64.b64encode(buffer2).decode('utf8')
    plt.clf()
    num = len(data)

    return render_template("image.html", image=pngImageB64String, image2=pngImageB64String_2, datas=data, len=num)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
