import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from libra.preprocessing.data_reader import DataReader

class ContentRecommender:
    '''
---------- Content Based Recommender System Class --------

This is the base class for the content based recommender. The constructor takes in 5 arguements:

data: the dataset to work on
feature_names = a list of the names of features you would like to use to get recommendations
n_recommendations: the number of recommendations to return
indexer = the name of the columns you want to get recommendations from

Example:
newClient.content_recommender_query(feature_names=['genre','actors','writer','plot'],indexer='title')

Methods:


recommend: the recommendations function. Returns recommendations based on search term passed
parameters:
search_term: string of the item you want to get recommendations from
returns:
result: a pandas DataFrame of the top n recommendations

'''



    def __init__(self,data=None,feature_names=[],n_recommendations=10,indexer='title'):
        dataReader = DataReader(data)
        self.data = dataReader.data_generator()
        self.feature_names = feature_names
        self.n_recommendations=n_recommendations
        self.indexer = indexer

        for f in self.feature_names:
            self.data[f] = self.data[f].apply(self.clean_data)

        self.data['soup'] = self.data.apply(self.create_soup,axis=1)   
        print(' |- Processing Data...')
        self.count_vec = CountVectorizer()

        # BOW and similarity matrix
        self.count_matrix = self.count_vec.fit_transform(self.data['soup'])
        self.sim_matrix = cosine_similarity(self.count_matrix,self.count_matrix)

        # mapping for the results
        self.data = self.data.reset_index()
        self.mapping = pd.Series(self.data.index, index=self.data[self.indexer])

    def clean_data(self,x):
        if isinstance(x, list):
            return np.array([str.lower(i.replace(" ", "")) for i in x if not x.isdigit()])
        else:
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    def create_soup(self,x):
        soup = []
        for feature in np.array(self.feature_names):
            f = ''.join(x[feature])
            soup.append(f)
        return ' '.join(soup)

    def recommend(self,s_name):
        self.index11 = self.mapping.loc[s_name]
        self.similarity_score = list(enumerate(self.sim_matrix[self.index11]))
        self.similarity_score = sorted(self.similarity_score, key=lambda x: x[1],reverse=True)
        self.similarity_score = self.similarity_score[1:self.n_recommendations]
        self.indices = [i[0] for i in self.similarity_score]
        return pd.DataFrame(self.data[self.indexer].iloc[self.indices]).reset_index().drop('index',axis=1)


