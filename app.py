from email import header
from operator import index
from flask import Flask, request, render_template, jsonify
from model import SentimentRecommenderModel


app = Flask(__name__)

sentiment_model = SentimentRecommenderModel()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['GET'])
def prediction():
    # get user from the html form
    #user = request.form['username']
    user = request.args.get("username")

    # convert text to lowercase
    user = user.lower()
    #print("Hi ", user)
    column_names=[]
    row_data=[]
    items=""
    items = sentiment_model.getSentimentRecommendations(user)
    
    if(not(items is None)):
        print(f"retrieving items....{len(items)}")
        print(items)
        # data=[items.to_html(classes="table-striped table-hover", header="true",index=False)
        #return render_template("index.html", column_names=items.columns.values, row_data=list(items.values.tolist()), zip=zip)
        return render_template("index.html",column_names=list(items.columns),row_data=items.values.tolist(),zip=zip)

    else:
        return render_template("index.html", message="User Name doesn't exists, No product recommendations at this point of time!")



if __name__ == '__main__':
    app.run()