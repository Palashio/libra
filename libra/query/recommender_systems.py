import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from colorama import Style
import networkx as nx
import math

counter = 0
currLog = ""

def clearLog():
    global counter
    global currLog

    currLog = ""
    counter = 0


# logging function that creates hierarchial display of the processes of
# different functions. Copied into different python files to maintain
# global variable parallels
def logger(instruction, found=""):
    '''
    logging function that creates hierarchial display of the processes of
    different functions. Copied into different python files to maintain
    global variables.

    :param instruction: what you want to be displayed
    :param found: if you want to display something found like target column
    '''
    
    global counter
    if counter == 0:
        print((" " * 2 * counter) + str(instruction) + str(found))
    elif instruction == "->":
        counter = counter - 1
        print(Fore.BLUE + (" " * 2 * counter) +
              str(instruction) + str(found) + (Style.RESET_ALL))
    else:
        print((" " * 2 * counter) + "|- " + str(instruction) + str(found))
        if instruction == "done...":
            print("\n" + "\n")

    counter += 1


# Cleaning/ preprocessing function for content based recommender engine
# basically converts feature values to list and fills in any NaN values
def preprocess(df,feature_names):
    for f in feature_names:
        df = df.copy()
        df[f] = df[f].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
        df = df.fillna('')
    return df



class ContentBasedRecommender:
    """ Main function for the Content Based Recommender """
    """
    Concretely, this returns the top n recommendations for the specified value based on given features.
    Parameters:

    dataset: dataset name
    feature_names: a list of features to be used to generate recommendations
    search_name: the name of the item you are looking to get recommendations from
    n_recommendations: number of recommendations(default=10)
    indexer: the name of the feature you want to get recommendations from
    text_feature: the feature that describes the item your are trying to recommend(description,plot,overview,etc)

    Example:

    For a movie dataset with features incliding genre,actors,writers:

    c.content_recommender_query(feature_names=['genre','actors','writer','director','country'],text_feature='plot')
    recommendations = c.recommend('Toy Story')

    Returns:
    a dictionary with the following keys:

    recommendations: the generated recommendations
    n_recommendations: the number of recommendations made
    indexer: the indexer used
    text_feature: the text feature used
    feature_names: the feature names used

    """
    def __init__(self,dataset,feature_names=[],text_feature=None,indexer='title',n_recommendations=10):
        self.df = preprocess(pd.read_csv(dataset),feature_names)
        logger('Preprocessing data...')
        if feature_names == []:
            # Getting defaults if no feature names were passed
            options = ['id','_id','rating','_rating','score','_score']
            self.feature_names = list(self.df.select_dtypes('object').columns)
            for i in options:
                if i in self.feature_names:
                    del self.feature_names[self.feature_names.index(i)]
        else:
            self.feature_names = feature_names
        if text_feature == '':
            opt = ['description','overview','plot','summary']
            for i in opt:
                if i in list(self.df.columns):
                    self.text_feature = i
                    break
        else:
            self.text_feature = text_feature

        # NetworkX Graph initialisation
        self.G = nx.Graph()
        self.indexer = indexer
        self.n_recommendations = n_recommendations
        
    def recommend(self,root):
        # TFIDF
        text_content = self.df[self.text_feature]
        vector = TfidfVectorizer(max_df=0.4,         
                                     min_df=1,      
                                     stop_words='english', 
                                     lowercase=True, 
                                     use_idf=True,   
                                     norm=u'l2',     
                                     smooth_idf=True 
                                    )
        tfidf = vector.fit_transform(text_content)

        # Mini-batch kmeans on tfidf
        k = 200
        kmeans = MiniBatchKMeans(n_clusters = k)
        kmeans.fit(tfidf)
        centers = kmeans.cluster_centers_.argsort()[:,::-1]
        terms = vector.get_feature_names()
        logger('Getting Bag of Words..')

        request_transform = vector.transform(text_content)
        self.df['cluster'] = kmeans.predict(request_transform) 

        # Find similiar indices        
        def find_similar(tfidf_matrix, index, top_n = 5):
            cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
            related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
            return [index for index in related_docs_indices][0:top_n] 
    
        for i, rowi in self.df.iterrows():
            self.G.add_node(rowi[self.indexer],label="ITEM")
            for f in self.feature_names:
                for element in rowi[f]:
                    self.G.add_node(element)
                    self.G.add_edge(rowi[self.indexer], element)

            indices = find_similar(tfidf, i, top_n = 5)
            logger('Getting cosine similarities..')
            snode="Sim("+rowi[self.indexer][:15].strip()+")"        
            self.G.add_node(snode,label="SIMILAR")
            self.G.add_edge(rowi[self.indexer], snode, label="SIMILARITY")
            for element in indices:
                self.G.add_edge(snode, self.df[self.indexer].loc[element], label="SIMILARITY")

        commons_dict = {}
        logger('Getting Recommendations..')
        for e in self.G.neighbors(root):
            for e2 in self.G.neighbors(e):
                if e2==root:
                    continue
                if self.G.nodes[e2]['label']=="ITEM":
                    commons = commons_dict.get(e2)
                    if commons==None:
                        commons_dict.update({e2 : [e]})
                    else:
                        commons.append(e)
                        commons_dict.update({e2 : commons})
        items=[]
        weight=[]
        for key, values in commons_dict.items():
            w=0.0
            for e in values:
                w=w+1/math.log(self.G.degree(e))
            items.append(key) 
            weight.append(w)

        result = pd.DataFrame(data=np.array(weight),index=items).reset_index().sort_values(0,ascending=False)
        result.columns = ['Title','Similarity'] 
        output = result[:self.n_recommendations]  
        logger("Done! Recommendations can be viewed as a DataFrame under the 'recommendations' key!") 
        return {
            'recommendations': output,
            'n_recommendations': self.n_recommendations,
            'indexer': self.indexer,
            'feature_names': self.feature_names,
            'text_feature': self.text_feature
        }