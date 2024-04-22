import os
import pandas as pd
import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
import string #to remove special characters and punctuation

def data_preperation(Emails_df):
    #print(Emails_df.columns) #['text', 'spam']
    #print(len(Emails_df))
    #drop empty rows
    Emails_df.dropna(inplace=True)
    #remove duplicated
    Emails_df.drop_duplicates(inplace=True)
    
    #Text normalization
    
    #covert text to lowercase
    Emails_df['text'] = Emails_df['text'].str.lower()
    
    #Tokenize text: change sentences to individual words. 
    #now 'text' will hold an array of individual words and not sentences
    #added a new column to the dataframe
    Emails_df['tokens'] = Emails_df['text'].apply(lambda x: x.split())
    
    #Remove stop words: "and/or/this/as.." these are common english
    #words that do not provide value to us
    
    #fetch the stopwords
    stop_words_filepath = os.path.join('resources', 'stopwords.csv')
   
    if not os.path.isfile(stop_words_filepath):
        print("Please run the download_resources.py script under the resources folder to download stopwords.")
        return None
    
    stop_words = set(pd.read_csv(stop_words_filepath)['stopword'])  #use set to get only unique words
    
    #function to remove stop words
    def remove_stopwords(tokens):
        tokens_without_stopwords= []
        
        for word in tokens:
            if word not in stop_words:
                tokens_without_stopwords.append(word) 
        return tokens_without_stopwords
   
    #When you use .apply() on a DataFrame column, it applies
    # the function to each element (each row) of that column 
    # individually. In this case, each element is a list of 
    # tokens, and you want to apply remove_stopwords to each 
    # of these lists separately.
    Emails_df['tokens'] = Emails_df['tokens'].apply(remove_stopwords)
    
    #stemming
    stemmer = PorterStemmer()
    #apply stemming to a list
    def apply_stemming(tokens):
        return [stemmer.stem(token) for token in tokens]
    
    Emails_df['tokens'] = Emails_df['tokens'].apply(apply_stemming)

    #Lemmatization
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    
    def apply_lem(tokens):
        return [lemmatizer.lemmatize(token) for token in tokens]
    
    Emails_df['tokens'] = Emails_df['tokens'].apply(apply_lem)
    
    #remove special characters and punctuation
    def remove_punctuation(tokens):
        #create translation table to map special characters and punctuation to none 
        map_to_none = str.maketrans('','',string.punctuation)
    
        #remove punctuation and special characters from each token
        cleaned_tokens = [token.translate(map_to_none) for token in tokens]
        
        #remove empty spaces
        cleaned_tokens = [token for token in cleaned_tokens if token]
        return cleaned_tokens
    Emails_df['tokens'] = Emails_df['tokens'].apply(remove_punctuation)
    #print(Emails_df.at[1, 'tokens'])
    return Emails_df


def start_here():
    emails_df = None
    
    try:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(current_dir, "resources/emails.csv") 
        emails_df = pd.read_csv(file_path) 
        emails_df = data_preperation(emails_df)
        print(emails_df.columns)
        # Save the processed DataFrame to a CSV file
        output_file_path = os.path.join(current_dir, "resources/processed_emails.csv")
        emails_df.to_csv(output_file_path, index=False)  # Set index=False to exclude row numbers
        return 
    
    except FileNotFoundError:
        print("File not found")
        
    except PermissionError:
        print("Permission denied")
        
    except Exception as e:
        print(f"Unexpected error:  {e}") 
                 
    return emails_df

start_here()

