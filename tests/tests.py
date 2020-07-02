from libra.queries import client

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

    newClient = client('../tools/data/structured_data/housing.csv')

    # Tests whether regression_ann_query works without errors, and creates a key in models dictionary
    @ordered
    def test_regression_ann(self):
        self.newClient.regression_query_ann('predict median house value', epochs=3)
        self.assertTrue(self.newClient.models.get('regression_ANN'))
        del self.newClient.models['regression_ANN']

    # Tests whether classification_ann_query works without errors, and creates a key in models dictionary
    @ordered
    def test_classification_ann(self):
        self.newClient.classification_query_ann('predict ocean proximity', epochs=3)
        self.assertTrue(self.newClient.models.get('classification_ANN'))
        del self.newClient.models['classification_ANN']

    # Tests whether neural_network_query uses the correct model
    @ordered
    def test_nn_query(self):
        # see if properly chooses regression with a numeric target column
        self.newClient.neural_network_query('predict median house value', epochs=3)
        self.assertTrue(self.newClient.models.get('regression_ANN'))

        # see if properly chooses classification with a categorical target column
        self.newClient.neural_network_query('predict ocean proximity', epochs=3)
        self.assertTrue(self.newClient.models.get('classification_ANN'))

    # Tests whether decision_tree_query works without errors, and creates a key in models dictionary
    @ordered
    def test_decision_tree(self):
        self.newClient.decision_tree_query('predict ocean proximity')
        self.assertTrue(self.newClient.models.get('decision_tree'))



unittest.main()