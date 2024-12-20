import random
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

class SVD_Recommender():
    def __init__(self, product_df):
        self.product_df = product_df
    
    def find_component(self, sdf):
        _, c = sdf.shape
        svd = TruncatedSVD(n_components=c, random_state=42)
        svd.fit(X)
    
        count=0
        for index, cumsum in enumerate(np.cumsum(svd.explained_variance_ratio_)):
            if cumsum<=0.95:
                count+=1
            else:
                break
        return count
        
    def fit_svd(self, sdf):
        svd = TruncatedSVD(n_components=self.find_component(sdf), random_state=random.randint(1,10))
        X = sdf.values
        matrix = svd.fit_transform(X)
        corr = np.corrcoef(matrix)    
        return corr
    
    def get_recommendation(self, sdf, user_list=None, num_users=5, num_products=5):
        if user_list:
            sample = user_list
        else:
             all_users = list(sdf.columns)
             sample = random.sample(all_users, num)
        
        all_restaurants = sdf.index
        rec_arr = self.fit_svd(sdf)
        rec_dict = {}
        for user in sample:
            idx = all_users.index(user)
            indices = np.argpartition(rec_arr[idx],num_products)[-num_products:]
            recs = list(all_restaurants[indices])
            rec_dict[f"user_{user.replace('-','')[:5]}"] = self.product_df[self.product_df['business_id'].isin(recs)]['name'].unique()
        return pd.DataFrame(rec_dict)      
