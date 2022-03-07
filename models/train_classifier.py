import sys
from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
#nltk.download('stopwords')
# List stop words from NLTK
from nltk.corpus import stopwords

import pandas as pd
from sqlalchemy import create_engine

nltk.download('wordnet') # download for lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

# Instantiate transformers and classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):

    # load data from database
    #path="..\\data\\DisasterResponse.db"
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM messages", con=engine)
    X = df.loc[:,"message"]
    Y = df.iloc[:,4:]
    category_names = Y.columns

    return X, Y, category_names


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


def build_model():
# Create a pipeline consists of vectorizer, tfidf transdormer and random forest cassifier with multiple output.

    # Create a pipeline 
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    ## Find the optimal model using GridSearchCV
    parameters = {
        'text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__weights': ['uniform', 'distance']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=5, cv=4, n_jobs=4)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Display the classificatipon report"""

    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """Save the  model into pickle file"""

    # Save the model based on model_filepath given
    filename = '{}'.format(model_filepath)
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()