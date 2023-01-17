# Disaster Response Pipeline Project
This project builds a website app that deploys a Natural Language Processing pipeline and a classification pipeline. The app has a text input box where one can type a text message. Once submitted, the app analyzes the input text and, in a fraction of a second, it returns a labelling binary output (yes/no, a green shadow = yes) for 36 request categories potentially relevant to disaster response management (i.e medical aid, food, water, shelter, fire, etc).

A quick demo:
![Disaster Response Pipeline Demo](img/demo.gif)

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

To run the app you can either visit the [website](url) where it is deployed or run it locally in your computer.

To run the app locally, you need python 3 and the following python modules installed: flask, plotly, nltk, sklearn, json, pandas, joblib, sqlalchemy, sys, sqlite3, re and pickle.

Once these modules are installed and the repository is cloned, to launch the app navigate to the directory called 'app' and run: `python run.py`. This will launch the Flask app and the terminal will show the url where it can be visited (typically [0.0.0.0:3001](http://0.0.0.0:3001/) or perhaps [127.0.0.1:3001](http://127.0.0.1:3001/))

If you need to re-build the database and train the classifier model, then run the following commands in the project's root directory. Please note that training the model is time-consuming (2-3 hours):

- To run the Extract-Transform-Load pipeline that cleans data and stores it in a database called 'DisasterResponse.db':
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run a Machine Learning pipeline that trains the classifier and saves the trained model as a Pickle file:
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

## Project Motivation<a name="motivation"></a>

The aim of the project is to build a classifier that quickly tags input text messages from social media posts into 36 possible categories to identify posts coming from people needing humanitarian aid during catastrophes such as earthquakes or others. The aim is to classify each input post text as to know to what various categories it can relate so to quickly connect the post message with particular organizations or response agents (i.e. NGOs providing medical assistance or other types of aid).

This project is the practical assignment of Udacity's Data Scientist nanodegree 'Data Engineering' course. 

## File Descriptions <a name="files"></a>

The root directory contains the README and LICENSE files and 3 directories called 'data', 'models' and 'app'.

The directory called 'data' contains the dataset and scripts/files related with the ETL stage:
- 2 csv files ('disaster_messages.csv' and 'disaster_categories.csv') that contain +27.000 social media text posts. This is the raw dataset that needs to be wrangled and processed to become ready to train the classifier (see below).
- A ETL pipeline python script ('process_data.py')
- The database with cleaned data that results from running the ETL pipeline ('DisasterResponse.db' or any other name typed when running the ETL pipeline from terminal as explained above)
- A jupyter notebook ('ETL pipeline preparation') showing an initial exploratory data analysis and explaining the data wrangling/cleaning process

The 'models' directory contains files and scripts related with building the machine learning classifier model, its training and evaluation:
- The machine learning pipeline script ('train_classifier') loads the data from the database ('DisasterResponse.db' or other name given when ruynning the ML pipeline from terminal, see above), builds and trains a classifier model, exports the trained model as a Pickle file named 'classifier.pkl' (or other name given in the terminal, as pointed above) and prints in the terminal the classification report for each category and each class including its precision, recall, f1-score and accuracy.
- The pickle file with the trained classifier model ('classifier.pkl' or other name given in the terminal when running the ML pipeline, see above)
- A jupyter notebook ('ML pipeline preparation') explaining the Natural Language Processing and model building, training, tuning and evaluation

The 'app' directory contains the html and Flask app ('run.py') files.

## Results<a name="results"></a>

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
See the License file. The author is Jose Viosca Ros who thanks Udacity for their support and guidance throughgout the Data Scientist nanodegree.
