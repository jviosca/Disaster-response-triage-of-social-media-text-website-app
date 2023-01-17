import sys
import sqlite3
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """ Function to load messages and categories csv files and merge them
    into 1 single dataframe

    Args:
        Paths to 'messages' and 'categories' csv files (strings)
    Returns:
        df (dataframe)
    """

    # Load datasets
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')
    # Merge datasets
    data = messages.merge(categories, how='left', on='id')
    # Split 'categories' into separate category columns
    categories = categories['categories'].str.split(';', expand=True)
    # Get column names from first row
    first_row = categories.iloc[0]
    category_colnames = [x[:-2] for x in first_row]
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1 (last character of the
    # string)
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column])
    # Replace categories column in data with newly created category columns from
    # 'categories'
    data.drop(columns=['categories'], inplace=True)
    data = data.merge(categories, how='left', on='id')
    return data


def clean_data(data):
    """ Function to remove duplicates, useless columns and convert all categories
    to binary in dataframe resulting from function 'load_data()'

    Args:
        Dataframe (data)
    Returns:
        Cleaned dataframe
    """

    # Remove duplicates
    data.drop_duplicates(subset=['message'], inplace=True)
    # Remove column 'child_alone' which does not reflect its intended meaning
    data.drop(columns=['child_alone'], inplace=True)
    # Remove value=2 from category 'related' to have all binary categories
    data = data.loc[data['related'] != 2]
    return data


def save_data(data, database_filename):
    """ Function to load cleaned dataset into a database

    Args:
        Cleaned dataframe (data)
        Name of database (string) where to load data
    Returns:
        None
    """

    conn = sqlite3.connect(database_filename)
    data.to_sql("messages_w_categories", conn, if_exists="replace", index=False)


def main():
    """ Function to run the entire data wrangling pipeline
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        data = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        data = clean_data(data)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(data, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
