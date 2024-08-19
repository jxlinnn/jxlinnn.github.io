import pandas as pd
import numpy as np
import re
import SVD_Recommender

df1 = pd.read_csv("yelp_review.csv")
df2 = pd.read_csv("yelp_business.csv")
# subset dataframe containing only data for restaurants
restaurant_df = df2[df2.loc[:, 'categories'].str.contains('food|restaruants', flags=re.IGNORECASE, na=False, regex=True)]

def update_cols(df):
  new = {}
  for col in restaurant_df.columns:
      if re.search('(attributes).+', col):
          t = col.split('.')
          new[col] = t[-1].lower()
  return new

restaurant_df = restaurant_df.rename(columns=new)

wanted_cols = ['business_id', 'name', 'city', 'state', 'stars']
sub_rest_df = restaurant_df.loc[:, wanted_cols]
sub_rest_df = sub_rest_df.rename(columns={'stars':'overall'})
other_cols = ['business_id', 'review_id', 'user_id', 'stars']
# merging user review and business(restaurant) dataframe
df = sub_rest_df.merge(df1[other_cols], on='business_id', how='inner')

# compute number of reviews by each user and select ones > 50
user_review_count = df.groupby('user_id')['stars'].count().sort_values(ascending=False)
most_active_users = user_review_count[user_review_count>50]

#find and select most popular/frequently visited restaurants based on review counts > 100
restaruant_review_count = df.groupby('business_id')['stars'].count().sort_values(ascending=False)
popular_restaurants = restaruant_review_count[restaruant_review_count>100]

restaurant_list = list(popular_restaurants.index)
user_list = list(most_active_users.index)
#generate dataframe for fitting
train_df = df[df['business_id'].isin(restaurant_list[:100]) & df['user_id'].isin(user_list[:30])]
df_pivot = train_df.pivot_table(index="business_id", columns="user_id", values="stars", aggfunc='first', fill_value=0)
sdf = df_pivot.astype(pd.SparseDtype("float", 0)) #sparse matrix is more efficient if deciding to expand dataset

#implement SVD estimator
svd = SVD_Recommendation(train_df)
recs_df = svd.get_recommendation(sdf)
