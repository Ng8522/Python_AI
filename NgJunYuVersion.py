#!/usr/bin/env python
# coding: utf-8

# # Music Reommender System 

# # Data Preparetion

# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from joblib import dump, load
import joblib

import warnings
warnings.filterwarnings('ignore')


# In[53]:


music_data = pd.read_csv('Spotify_Youtube.csv', encoding='utf-8')
music_data.head()


# In[54]:


#find missing value
music_data.isnull().sum()


# In[55]:


#clear the data with missing values
music_data.dropna(inplace=True)
music_data.isnull().sum()


# In[56]:


#check duplicate data
music_data.duplicated().sum()


# In[57]:


#delete duplicate data(in this dataset don't have so dismiss)
music_data=music_data.drop_duplicates()
music_data.duplicated().sum()


# In[58]:


#(row, column)
music_data.shape


# In[59]:


music_data['Artist']


# # Data Exploration

# In[60]:


music_data.describe()


# In[61]:


views = music_data['Views']
total_views = views.sum()
print("The total views track in the songs list is: ", int(total_views))
max_views = views.max()
print("The most views track in the songs list is: ", int(max_views))
min_views = views.min()
print("The least views tracks in the songs list is: ", int(min_views))
mean_views = views.mean()
print("The mean views tracks in the songs list is: " , round(mean_views, 4))
median_views = np.median(views)
print("The median views tracks in the songs list is: ", round(median_views,4))
std_views = np.std(views)
print("The standard deviation likes tracks in the songs list is: ", round(std_views,4))


# In[62]:


likes = music_data['Likes']
total_likes = likes.sum()
print("The total likes that given by the users is: ", int(total_likes))
max_likes = likes.max()
print("The most likes track in the songs list is: ", int(max_likes))
min_likes = likes.min()
print("The least likes tracks in the songs list is: ", int(min_likes))
mean_likes = likes.mean()
print("The mean likes tracks in the songs list is: " , round(mean_likes, 4))
median_likes = np.median(likes)
print("The median likes tracks in the songs list is: ", round(median_likes,4))
std_likes = np.std(likes)
print("The standard deviation views tracks in the songs list is: ", round(std_likes,4))


# In[63]:


comments = music_data['Comments']
total_comments = comments.sum()
print("The total comments leave by the users is: ", int(total_comments))
max_comments = comments.max()
print("The most comments track in the songs list is: ", int(max_comments))
min_comments = comments.min()
print("The least comments tracks in the songs list is: ", int(min_comments))
mean_comments = comments.mean()
print("The mean comments tracks in the songs list is: " , round(mean_comments, 4))
median_comments = np.median(comments)
print("The median comments tracks in the songs list is: ", round(median_comments,4))
std_comments = np.std(comments)
print("The standard deviation comments tracks in the songs list is: ", round(std_comments,4))


# # Data Visualization 

# In[64]:


normalized_total_views = (total_views - min_views) / (max_views - min_views)
normalized_total_likes = (total_likes - min_likes) / (max_likes - min_likes)
normalized_total_comments = (total_comments - min_comments) / (max_comments - min_comments)

categories = ['Views', 'Likes', 'Comments']
normalized_totals = [normalized_total_views, normalized_total_likes, normalized_total_comments]

plt.figure(figsize=(8, 8))
plt.pie(normalized_totals, labels=categories, autopct='%1.1f%%', colors=['blue', 'green', 'orange'])
plt.title('Min-Max Normalization of Total Views, Likes and Comments')
plt.show()


# In[65]:


categories = ['Views', 'Likes', 'Comments']
totals = [total_views, total_likes, total_comments]

mean_totals = sum(totals) / len(totals)
std_totals = np.std(totals)

normalized_totals = [(x - mean_totals) / std_totals for x in totals]

plt.figure(figsize=(8, 6))
plt.bar(categories, normalized_totals, color=['blue', 'green', 'orange'])
plt.axhline(0, color='gray', linewidth=0.5)  # Add horizontal line at y=0
plt.title('Standard Deviation Normalized Total Views, Likes and Comments')
plt.xlabel('Categories')
plt.ylabel('Normalized Values')
plt.show()


# #  Deployment

# In[69]:


def collaborative_filtering(track_title):
    similarity_matrix = cosine_similarity(music_data[['Likes', 'Views', 'Comments']])
    
    track_index = music_data[music_data['Track'] == track_title].index[0]
    
    similar_tracks = list(enumerate(similarity_matrix[track_index]))
    
    sorted_similar_tracks = sorted(similar_tracks, key=lambda x: x[1], reverse=True)
    
    top_similar_tracks = sorted_similar_tracks[1:11]
    
    # Return top similar tracks
    return top_similar_tracks

# Testing
track_title = "Feel Good Inc."
collab_filtering_result = collaborative_filtering(track_title)
print("Top 10 tracks similar to", track_title, "based on Collaborative Filtering:")

for track in collab_filtering_result:
    print("- Track:", music_data.iloc[track[0]]['Track'])
    print("  Similarity Score:", track[1])

dump(collab_filtering_result, 'music_recommender_model_collaborative.joblib')
print("Collaborative filtering model saved successfully.")

dump(collab_filtering_result, 'collab_filtering_result.joblib')


# In[ ]:


import difflib

def find_closest_match(user_input):
    track_titles = music_data['Track'].tolist()
    
    closest_matches = difflib.get_close_matches(user_input, track_titles, n=1, cutoff=0.6)
    
    if closest_matches:
        return closest_matches[0]
    else:
        return None
    
track_title = input("Enter the title of the track: ")

closest_match = find_closest_match(track_title)

if closest_match:
    print("Closest match found:", closest_match)
    collab_filtering_result = collaborative_filtering(closest_match)
    print("\nTop 10 tracks similar to", closest_match, "based on Collaborative Filtering:")
    for track in collab_filtering_result:
        print(music_data.iloc[track[0]]['Track'])
else:
    print("There's no track such as", track_title, "Please enter another title")


# In[ ]:





# In[ ]:




