import os

class edaDashboard(object):
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data

    def dashboard(self):
        os.system("streamlit run /libra/dashboard/LibEDA.py "+self.path_to_data)

