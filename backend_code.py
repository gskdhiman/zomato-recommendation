# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 00:14:39 2020

@author: Gursewak
"""

import pandas as pd
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
data_path = 'data.csv'
df = pd.read_csv(data_path)
N = 3  # maximum recommendations

cost_for_two = 'approx_cost(for two people)'
location = 'listed_in(city)'
listing_type = 'listed_in(type)'
listing_city = 'listed_in(city)'
online_order = 'online_order'

# making cost of two as float
df[cost_for_two]=df[cost_for_two].str.replace(",",'').astype(float)

def create_knn():
    STOPWORDS = set(stopwords.words('english'))
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    def clean_data(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
        return url_pattern.sub(r'', text)
    
    df["reviews_list"] = df["reviews_list"].apply(lambda x: clean_data(x))
    
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    
    corpus = df['reviews_list'].tolist()
    tfidf_matrix = tfidf.fit_transform(corpus )
    
    knn_recomm = NearestNeighbors(metric = 'cosine', algorithm = 'brute',n_neighbors=30)
    knn_recomm.fit(tfidf_matrix)
    return knn_recomm,tfidf

knn_recomm,tfidf = create_knn()

def restaurant_recommend(user_input_text,budget,location,cuisine_type):
    start_time = datetime.now() 
    user_inp_mat = tfidf.transform([user_input_text])    
    # user_inp_mat.shape
    score,idx = knn_recomm.kneighbors(user_inp_mat.reshape(1, -1))
    score_idx = dict(zip(idx[0],score[0]))
    df_user = df.iloc[idx[0]]
    
    df_loc  = df_user
    if location is not None:
        df_loc = df_user[df_user['location'].str.lower().str.contains(location.lower())]
    
    df_budget = df_loc
    if budget is not None:
        df_budget = df_loc[df_loc[cost_for_two] <= budget]
    
    df_cuisine = df_budget
    if cuisine_type is not None:
        df_cuisine = df_budget[df_budget['cuisines'].str.lower().str.contains(cuisine_type.lower())]
    
    final_recommend = {}
    for idx,row in df_cuisine.iterrows():
        rest_name = row['name']
        score = score_idx[idx]
        score = str(round(score, 2)*100)+" %"
        final_recommend[rest_name] = score 
    
    final_recommend = sorted(final_recommend.items(), key=lambda x: x[1], reverse=True)
    final_recommend = final_recommend[:N]
    recomendation_time = (datetime.now() -start_time).seconds
    return final_recommend,recomendation_time 
    


# restaurant_recommend(user_input_text = 'Lassi and paratha',
#                      budget = 1000,
#                      location = 'Koramangala',
#                      cuisine_type= 'north indian')


# restaurant_recommend(user_input_text = 'good ambiance restaurants, serving fish',
#                      budget = None,
#                      location = 'Koramangala',
#                      cuisine_type= None)

# restaurant_recommend(user_input_text = 'must visit restaurants',
#                      budget = 1000,
#                      location = None,
#                      cuisine_type= 'north indian')


# restaurant_recommend(user_input_text = 'best cakes',
#                      budget = 1000,
#                      location = 'Koramangala',
#                      cuisine_type= None)


# restaurant_recommend(user_input_text = 'authentic chicken biryani',
#                      budget = 800,
#                      location = 'BTM',
#                      cuisine_type= None)
