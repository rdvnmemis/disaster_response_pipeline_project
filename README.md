# Disaster Response Pipeline Project


## Prerequisites
### Libraries
In order to use this model flowlesly you need to instakl the required librarier. Run the below code to install the related libraries
```
pip install -r requirements.txt
```
### Model
In `train_classifier.py' file i used KNeighborsClassifier , you can try different classifeirs to increase the accuracy.

---

## Introduction
In this project, we created a machine learning pipilene classifying messages coming from people in case of a disaster. The model classifies the messages to decrease the burden on officials.
There are three major steps in this project. When we run the model it reald the messages in the previously creted databas and classify the message that the user enter in the web app.
If the user want to use another train data, then by using "proces_data.py" one can create anptheher database table. Also, the user can use another classification algoith to train the data by modifying
"train_classifier.py" file. In order to classify a new message  go to "go" page (http://192.168.1.2:3001/go) and type the message to classify it.


---

## Files

#### `data/disaster_categories.csv` , `data/disaster_messages.csv`

In this projects we have two csv file contaninig info to be merged, `disaster_messages.csv` contains the messages, the messages are originally either in English or in other languages, all the messages have English tranlation. `disaster_categories.csv` contains the categories of each message for each message.
The two csv file can be merged by using the id column.

#### `data/process_data.py`

ETL process is driven in this file, here we merhe two csv file and clean the data before storing in the SqlAlchemy database table (messages) 

#### `models/train_classifier.py`

This is where we train our model to obtain the classfier. We load the data from the sqlite database, do some text preprocessing using Count Vectorizer and TF-IDF, train the model, and save the model as a pickle object.

#### `app/run.py`

The flask application that being used to run the web app. You can modify the web page in the `app/templates/master.html` and `app/templates/go.html`.

---

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run ML pipeline that trains classifier and saves
        ```
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the app's directory to run your web app. 
```
python run.py
```

3. Go to http://0.0.0.0:3001/
