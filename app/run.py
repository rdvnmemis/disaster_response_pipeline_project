import nltk
import json
import plotly
import pandas as pd
import plotly.graph_objects as go

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt','wordnet'])

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine
import plotly.graph_objs as go


app = Flask(__name__)


def tokenize(text):
        """ This function takes aa text and apply the NLP procedures (tokenize,removal of dtop words, lammatization)
    
        INPUT: 
            text:String 
        OUTPUT:
            words: list, a list constritutes of words 
        """
        
        # Normalize text
        text = text.lower()
        # Tokenize text
        words = word_tokenize(text)
        # Reduce words to their root form
        lemmatized_tokens = [WordNetLemmatizer().lemmatize(w) for w in words]
        
        return lemmatized_tokens

# create figues to pass to index funtion
def return_figures(df):
    """Creates three plotly visualizations

    Args:
        df: DataFrame, input data file

    Returns:
        list (dict): list containing the four plotly visualizations

    """


    graph_one = []
    #create a pandas series indicating the the number of messages for each text lengh less than 100
    series1 = df[df['text length']<100].groupby('text length').count()['id']

    graph_one.append(
      Bar(
      x = series1.index,
      y = series1.values,
      )
    )

    layout_one = dict(title = 'Distribution of message lengths in term of words',
                xaxis = dict(title = '# of words',),
                yaxis = dict(title = '# of messages'),
                )


    graph_two = []
    # calculate the number of messages in each genre 
    series2 = df.groupby('genre').count()['id'].sort_values()

    graph_two.append(
      Bar(
      x = series2.values,
      y = series2.index,
      orientation='h'
      )
    )
  

    layout_two = dict(title = 'classifification of Messages (Genre)',
                xaxis = dict(title = 'Counts',),
                yaxis = dict(title = 'Genre'),
                )


    
    # append all charts to the figures list


    graph_three = []
    # calculate total messages per category and take the top 10 
    series3 = df.drop(columns=['id','message','original','genre', 'text length']).sum().sort_values(ascending=False).head(10)

    graph_three.append(
      Bar(
      x = series3.index,
      y = series3.values,
      )
    )

    layout_three = dict(title = 'Total Messages per Category (Top 10)',
                xaxis = dict(title = 'Category'),
                yaxis = dict(title = '# of messages'),
                )

    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))

    return figures

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    df['text length'] = df['message'].apply(lambda x: len(x.split()))
    figures = return_figures(df)

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=figuresJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
    #app.run(host='localhost:5000', debug=True)

if __name__ == '__main__':
    main()
