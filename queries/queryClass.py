class queryHistory: 
    def __init__(self, data):    
        self.plots = {}
        self.models = {}
        self.data = data

    def printPlotList(self):
        print(plots)

    def printPlots(self):
        for plt in plots.values():
            plt.show()


def getPlot(the_object, plot_name):
    return the_object.plots[str(plot_name)]

