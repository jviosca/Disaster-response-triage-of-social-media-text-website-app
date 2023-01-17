import sys
import sqlite3
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    """ Function to load messages and categories csv files and merge them into 1 single dataframe
    
    Args:
        Paths to 'messages' and 'categories' csv files (strings)
    Returns:
        df (dataframe)
    """
    
    # Load datasets
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')
    # Merge datasets
    df = messages.merge(categories, how='left', on='id')
    # Split 'categories' into separate category columns
    categories = categories['categories'].str.split(';',expand=True)
    # Get column names from first row
    first_row = categories.iloc[0]
    category_colnames = [x[:-2] for x in first_row]
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1 (last character of the string)
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column])
    # Replace categories column in df with newly created category columns from 'categories'
    df.drop(columns=['categories'], inplace=True)
    df = df.merge(categories, how='left', on='id')
    return df

def clean_data(df):
    """ Function to remove duplicates, useless columns and convert all categories to binary in dataframe resulting from function 'load_data()'
    
    Args:
        Dataframe (df)
    Returns:
        Cleaned dataframe
    """
    
    # Remove duplicates
    df.drop_duplicates(subset=['message'], inplace=True)
    # Remove column 'child_alone' which does not reflect its intended meaning
    df.drop(columns=['child_alone'], inplace = True)
    # Remove value=2 from category 'related' to have all binary categories
    df = df.loc[df['related']!=2]
    return df

def save_data(df, database_filename):
    """ Function to load cleaned dataset into a database
    
    Args:
        Cleaned dataframe (df)
        Name of database (string) where to load data
    Returns:
        None
    """
    
    conn = sqlite3.connect(database_filename)
    df.to_sql("messages_w_categories", conn, if_exists="replace", index=False)


def main():
    """ Function to run the entire data wrangling pipeline
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
