
#The IMDb 5000 Movie Dataset available on Kaggle 
# is derived from data on the IMDb website, 
# which is one of the most popular and widely used sources 
# of movie information on the internet.
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import numpy as np

#'plot_keywords'
#'movie_title'
#'imdb_score'


def plot_box_whiskers(movies_df):
    
    #get the quartiles and min and max values
    Q1 = np.percentile(movies_df['imdb_score'],25)
    Q2 = np.percentile(movies_df['imdb_score'],50)
    Q3 = np.percentile(movies_df['imdb_score'],75)
    
    
    """plt.figure(figsize=(8,6))
    plt.boxplot(movies_df['imdb_score'], vert=False)
    plt.title('IMDb scores distribution')
    plt.xlabel("IMDb scores")  
    plt.show()"""
    
    return Q1, Q2, Q3

def data_preperation(movies_df):
    #print(movies_df.columns) #output is: Index(['color', 'director_name', 'num_critic_for_reviews', 'duration','director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',  'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name', 'movie_title', 'num_voted_users', 'cast_total_facebook_likes','actor_3_name', 'facenumber_in_poster', 'plot_keywords', 'movie_imdb_link', 'num_user_for_reviews', 'language', 'country','content_rating', 'budget', 'title_year', 'actor_2_facebook_likes','imdb_score', 'aspect_ratio', 'movie_facebook_likes'],dtype='object')    
    
    #drop rows with null values
    movies_df.dropna(inplace=True) #1288 rows with null values
    #reomve duplicates
    movies_df.drop_duplicates(inplace=True) #33 duplicates
    
    #keep 'plot_keywords', 'movie_title' 'imdb_score'
    movies_df = movies_df[['movie_title','plot_keywords','imdb_score']].copy() #use a copy going forward and not a view
    
    Q1, Q2, Q3 = plot_box_whiskers(movies_df)
    
    #Using these statistical values the classes for each movie will be
    #derived as following:
    #Bad Movies (low quality): imdb_scores below Q1 (25th percentile).
    #Average Movies (decent quality): imdb_scores between Q1 and Q3 (interquartile range).
    #Good Movies (good quality): IMDb scores above Q3 (75th percentile)
    
    quality_categories = []
    
    for score in movies_df['imdb_score']:
        if score < Q1:
           quality_categories.append('Bad')
           
        elif Q1 <= score <= Q2:
            quality_categories.append('Average')
        else:
            quality_categories.append('Good')

    #Add the new column to the dataframe
    #movies_df.drop(columns=['quality'], inplace=True)
    #movies_df.loc[:, 'quality'] = quality_categories
    movies_df.loc[:, 'quality'] = quality_categories

    #print(len(quality_categories))  
    #print(movies_df['plot_keywords'])  
    
    return movies_df


def start_here():
    movies_df = None
    try:
        # Get the current directory of the Python script
        current_dir = os.path.dirname(os.path.realpath(__file__))

        file_path = os.path.join(current_dir, "movie_metadata.csv")
        movies_df = pd.read_csv(file_path)
        return data_preperation(movies_df)
    
    except FileNotFoundError: #handilng missing file
        print("Error: File not found.")
    except PermissionError: #handling any locked files
        print("Error: Permission denied.")
    except Exception as e: #any other error
        print("An unexpected error occurred:", e)
        
        return movies_df

#uncomment me below:
#start_here()
