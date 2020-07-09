from libra import client

import unittest

def make_orderer():
    order = {}

    def ordered(f):
        order[f.__name__] = len(order)
        return f

    def compare(a, b):
        return [1, -1][order[a] < order[b]]

    return ordered, compare


ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare

class TestQueries(unittest.TestCase):

    newClient = client('tools/data/structured_data/housing.csv')

    """
    TEST QUERIES
    
    Tests some queries in queries.py
    """
    # Tests whether regression_ann_query works without errors, and creates a key in models dictionary
    @ordered
    def test_regression_ann(self):
        self.newClient.regression_query_ann('predict median house value', epochs=3)
        self.assertTrue('regression_ANN' in self.newClient.models)
        del self.newClient.models['regression_ANN']

    # Tests whether classification_ann_query works without errors, and creates a key in models dictionary
    @ordered
    def test_classification_ann(self):
        self.newClient.classification_query_ann('predict ocean proximity', epochs=3)
        self.assertTrue('classification_ANN' in self.newClient.models)
        del self.newClient.models['classification_ANN']

    # Tests whether neural_network_query uses the correct model
    @ordered
    def test_nn_query(self):
        # see if properly chooses regression with a numeric target column
        self.newClient.neural_network_query('predict median house value', epochs=3)
        self.assertTrue('regression_ANN' in self.newClient.models)

        # see if properly chooses classification with a categorical target column
        self.newClient.neural_network_query('predict ocean proximity', epochs=3)
        self.assertTrue('classification_ANN' in self.newClient.models)

    # Tests whether decision_tree_query works without errors, and creates a key in models dictionary
    @ordered
    def test_decision_tree(self):
        self.newClient.decision_tree_query('predict ocean proximity')
        self.assertTrue('decision_tree' in self.newClient.models)

    # Tests whether svm_query works without errors, and creates a key in models dictionary
    @ordered
    def test_svm(self):
        self.newClient.svm_query('predict ocean proximity')
        self.assertTrue('svm' in self.newClient.models)

    # Tests whether nearest_neighbor_query works without errors, and creates a key in models dictionary
    @ordered
    def test_nearest_neighbors(self):
        self.newClient.nearest_neighbor_query('predict ocean proximity')
        self.assertTrue('nearest_neighbor' in self.newClient.models)

    # Tests whether kmeans_clustering_query works without errors, and creates a key in models dictionary
    @ordered
    def test_kmeans(self):
        self.newClient.kmeans_clustering_query('predict ocean proximity')
        self.assertTrue('k_means_clustering' in self.newClient.models)


    """
    TEST ANALYZE() FUNCTION
    
    Tests all branches of .analyze() function in generate_plots
    """
    # Tests analyze() function for k_means_clustering
    @ordered
    def test_analyze_kmeans(self):
        self.newClient.analyze(model='k_means_clustering')
        self.assertTrue('n_centers' in self.newClient.models['k_means_clustering'])
        self.assertTrue('centroids' in self.newClient.models['k_means_clustering'])
        self.assertTrue('inertia' in self.newClient.models['k_means_clustering'])

    # Tests analyze() function on regression_ANN
    @ordered
    def test_analyze_regression(self):
        self.newClient.analyze(model='regression_ANN')
        self.assertTrue('MSE' in self.newClient.models['regression_ANN'])
        self.assertTrue('MAE' in self.newClient.models['regression_ANN'])

    # Tests analyze() function on classification_ANN
    @ordered
    def test_analyze_classification(self):
        self.newClient.analyze(model='classification_ANN')
        self.assertTrue('plots' in self.newClient.models['classification_ANN'])
        self.assertTrue('roc_curve' in self.newClient.models['classification_ANN']['plots'])
        self.assertTrue('confusion_matrix' in self.newClient.models['classification_ANN']['plots'])

        self.assertTrue('scores' in self.newClient.models['classification_ANN'])
        self.assertTrue('recall_score' in self.newClient.models['classification_ANN']['scores'])
        self.assertTrue('precision_score' in self.newClient.models['classification_ANN']['scores'])
        self.assertTrue('f1_score' in self.newClient.models['classification_ANN']['scores'])

    # Tests analyze() function for sklearn models
    @ordered
    def test_analyze_sklearn_classifiers(self):
        for mod in ['svm', 'nearest_neighbor', 'decision_tree']:
            self.newClient.analyze(model=mod)
            modeldict = self.newClient.models[mod]

            self.assertTrue('plots' in modeldict)
            self.assertTrue('roc_curve' in modeldict['plots'])
            self.assertTrue('confusion_matrix' in modeldict['plots'])

            self.assertTrue('scores' in modeldict)
            self.assertTrue('recall_score' in modeldict['scores'])
            self.assertTrue('precision_score' in modeldict['scores'])
            self.assertTrue('f1_score' in modeldict['scores'])

    # Tests invalid model input
    @ordered
    def test_invalid_model(self):
        with self.assertRaises(NameError):
            self.newClient.analyze(model='I dont exist')

if __name__ == '__main__':
    unittest.main()
