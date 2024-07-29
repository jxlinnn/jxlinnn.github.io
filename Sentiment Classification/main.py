import json
import csv
import pandas as pd
import numpy as np
import nltk
import re
from data_cleaning import *

df1 = pd.read_csv("yelp_review.csv")
df2 = pd.read_csv("yelp_business.csv")

sub_df1 = df1[['review_id', 'business_id', 'stars', 'text']]
sub_df2 = df2[df2.loc[:, 'categories'].str.contains('food|restaruants', flags=re.IGNORECASE, na=False, regex=True)]
restaurant_df = sub_df1[sub_df1.business_id.isin(sub_df2.business_id.unique().tolist())]

text_tuple = wrap_text(restaurant_df.text, restaurant_df.stars)
train_df = convert_to_csv(text_tuple, columns=['review','rating'])
review_rating = train_df.values.tolist()


                                               
