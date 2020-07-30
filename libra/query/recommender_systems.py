import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from libra.preprocessing.data_reader import DataReader
from colorama import Style

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
# basically tokenizes all the feature examples
def clean_data(x):
    if isinstance(x, list):
        return np.array([str.lower(i.replace(" ", "")) for i in x if not x.isdigit()])
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# Cosine similarity matrix creator
def matrix_maker(data,indexer='title',feature_names=[]):
    try:
        assert(isinstance(feature_names, list))
    except:
        logger('Error! Feature names must be of type list!')
        exit()


    # function to combine all the tokenized features values for the
    # cosine matrix to calculate similarities from (the recommender 'soup')
    def create_soup(x):
        soup = []
        for feature in np.array(feature_names):
            f = ''.join(x[feature])
            soup.append(f)
        return ' '.join(soup)

    data = pd.read_csv(data)
    data = data.copy()
    #data = data[['title']].apply(pd.Series.unique).join(data[feature_names])
    try:
        assert indexer in data.columns
    except:
        logger('Error! the indexer passed named:  ' +  str(indexer) + 'is not in dataset!')
        exit()
    for f in feature_names:
            data[f] = data[f].apply(clean_data)
    logger('Cleaning Data..')

    data['text'] = data.apply(create_soup,axis=1)
    logger('Getting similarities')

    # Create a CountVectorizer, fit to data 'soup' and get similarities
    con = CountVectorizer()
    item_matrix = con.fit_transform(data['text'])
    cosine_similarities = cosine_similarity(item_matrix,item_matrix)
    similarities = {}

    # Loop through similarities and get top 50, return similarities
    for i in range(len(cosine_similarities)):
        similar_indices = cosine_similarities[i].argsort()[:-50:-1] 
        similarities[data[indexer].iloc[i]] = [(cosine_similarities[i][x], data[indexer][x], '') for x in similar_indices][1:]
    return similarities

class ContentBasedRecommender:
    '''
    ---------- Content Based Recommender System Class --------

    This is the base class for the content based recommender. The constructor takes in 5 arguements:

    data: the dataset to work on
    feature_names = a list of the names of features you would like to use to get recommendations
    by default this will be all the categorical columns
    n_recommendations: the number of recommendations to return
    indexer = the name of the columns you want to get recommendations from
    by default this will be the first categorical columns(excluding the id)

    Example:
    newClient.content_recommender_query(feature_names=['genre','actors','writer','plot'],indexer='title')

    Methods:

    recommend: the recommendations function. Returns recommendations based on search term passed
    parameters:
    search_term: string of the item you want to get recommendations from
    returns:
    result: a pandas DataFrame of the top n recommendations

    Example:
    c = client('path to file')
    c.content_recommender_query(feature_names=['genre','plot','director','actors','writer'])
    recommendations = c.recommend('Coco')


    _get_message: gets the results of the similarity and creates a Dataframe
    of the resultd and their correlation.

    '''


    def __init__(self, data,feature_names=[],indexer='',n_recommendations=10):
        # If feature names is blank, then it get all categorical objects,
        # removes the id and used them to recommend items,setting the indexer
        # as the first element of the feature_names
        dataReader = DataReader(data)
        data1 = dataReader.data_generator()
        self.data1 = data1.copy()

        if feature_names == []:
            catnames = list(self.data1.select_dtypes('object').columns)
            for i in catnames:
                if 'id' in i.lower():
                    v = catnames.index(i)
                    del catnames[v]
            self.feature_names = catnames[1:]
            self.indexer = catnames[0]
        else:
            # Initialise default variables
            self.indexer = indexer
            self.feature_names = feature_names

        # call the matrix_maker function created above
        self.matrix_similar = matrix_maker(data,self.indexer,self.feature_names)
        self.n_recommendations=n_recommendations

    def _get_message(self, item, recom_items):
        # Get the recommendations and put them into a DataFrame
        logger("Complete! Stored recommendation DataFrame under the 'recommendations' key")
        clearLog()

        rec_items = len(recom_items)
        # List for recommended items and their respective correlations
        recommended_items = []
        recommended_corr = []

        # Loop through each item,append to the correct list, and returnt a dict key
        # of the DataFrame,n_recommendations, feature_names and indexer
        for i in range(rec_items):
            recommended_items.append(recom_items[i][1])
            recommended_corr.append(round(recom_items[i][0], 3))

        
        df = pd.DataFrame(pd.Series(np.array(recommended_items)),columns=['Recommendations'])
        df.insert(1,'Correlation',recommended_corr)


        return {
            'recommendations': df,
            'indexer': self.indexer,
            'n_recommendations': self.n_recommendations,
            'feature_names': self.feature_names
        }

        
    # recommendation function
    def recommend(self, s_name):
        # Get item to find recommendations for
        item = s_name
        # Get number of items to recommend
        number_items = self.n_recommendations
        # Get the number of items most similars from matrix similarities
        recom_item = self.matrix_similar[item][:number_items]
        # return each item

        return self._get_message(item=item, recom_items=recom_item)
