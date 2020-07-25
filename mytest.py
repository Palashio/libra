from libra import client
import pandas as pd
c = client('/Users/Vagif/Downloads/disney_plus_shows.csv')
c.content_recommender_query(feature_names=['genre','plot','director'],indexer='title')
qw = c.recommend('Toy Story')
print(qw)
