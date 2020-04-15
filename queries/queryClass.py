class queryHistory: 
    def __init__(self, data):    
        self.plots = {}
        self.models = {}
        self.data = data


def getPlot(the_object, plot_name):
    return the_object.plots[str(plot_name)]

