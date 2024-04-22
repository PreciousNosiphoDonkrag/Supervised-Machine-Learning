import nltk
from nltk.corpus import stopwords #allows us to download the stopwords that we can use for data preperation 
import pandas as pd
import os

stopwords_file_path = os.path.join('resources', 'stopwords.csv')

if not os.path.isfile(stopwords_file_path):
    # Download 
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    
    stopwords_df = pd.DataFrame({'stopword': stop_words})
    stopwords_df.to_csv(stopwords_file_path, index=False)
else:
    print("Stopwords file already exists.")
    
#wordnet
nltk.data.path.append('resources/nltk_data')
wordnet_path = os.path.join('resources', 'nltk_data', 'corpora', 'wordnet')

if not os.path.exists(wordnet_path):
    print("WordNet not found. Downloading...")
    nltk.download('wordnet', download_dir='resources/nltk_data')
else:
    print("Wordnet already exists.")

