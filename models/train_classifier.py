import sys
import re
import pickle
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'stopwords', 'wordnet'])


def load_data(database_filepath):
    """ Function to get data from a database and extract feature (X) and target (y) variables.
    It also extracts category names from the dataset

    Args:
        database filepath (string)
    Returns:
        Features variable (pandas series): X
        Values of target variables (pandas dataframe): y
        Names of categories (list of strings): category_names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    data = pd.read_sql('select * from messages_w_categories', engine)
    X = data['message']
    y = data.drop(columns=['message', 'original', 'genre'])
    category_names = y.columns.values.tolist()
    return X, y, category_names


def tokenize(text):
    """Function to clean, normalize, tokenize and lemmatize text

    Args:
        Text to process (string): text
    Returns:
        Processed text (list of strings): clean_tokens
    """
    # remove urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Remove spaces > 1
    text = re.sub(' +', ' ', text)
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and remove stop words
    clean_tokens = [WordNetLemmatizer().lemmatize(w)
                    for w in tokens if w not in stopwords.words("english")]
    return clean_tokens


def build_model():
    """ Function to build a logistic regression classifier of text messages
    on 36 categories or labels.

    GridSearch is used to optimize for the use of balanced class weights
    vs not using class weight balance.

    In balanced class weight, weights are inverserly proportional to class
    frequencies in input data as:
        n_samples / (n_classes * np.bincount(y)))
    In class weight = None, all classes are supposed to have weight = 1.

    f1_macro is used as scorer as it is more sensitive to the score of the
    less frequent class, which in this use case
    is generally preferred due to high class imbalance in many categories/labels.

    Args:
        None
    Returns:
        Classification model (skilearn pipeline object): cv
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        # default max_iter=100 raises warning 'STOP: TOTAL NO. of ITERATIONS
        # REACHED LIMIT'
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=200)))
    ])

    # specify parameters for grid search
    parameters = {
        'clf__estimator__class_weight': [None, 'balanced']
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_macro')
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Function to print 1 classification report per each category/label
    with all its classes.
    In each report of each label, includes:
    - Precision, recall and F1 for each class of each label in y_test.
    - Weighted and unweighted (macro) averages of precision, recall and F1.
    - Accuracy (= unweighted recall).

    Args:
        Classification model (skilearn pipeline object): model
        Test split of feature variable (pandas series): X_test
        Test split of target variables/labels (pandas dataframe): Y_test
        Names of target variables (list of strings): category_names
    Returns:
        None
    """

    predicted = model.predict(X_test)
    for i, column in enumerate(category_names):
        print("****\ncategory: " + column)
        category_values = Y_test[column].value_counts().index
        print(classification_report(Y_test[column], predicted[:, i], target_names=[
              'value: ' + str(i) for i in category_values]))


def save_model(model, model_filepath):
    """ Function to export classification model as a pickle file

    Args:
        Classification model (skilearn pipeline object): model
        Path of file to create (string): model_filepath
    Returns:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Function to run the entire machine lerning text classification
    pipeline, from loading the dataset, training the classifier to
    exporting the model as a pickle file.
    """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
