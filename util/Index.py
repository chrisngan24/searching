import pickle
import os

class Index:
    def __init__(self):
        self.index = {}

    def put(self,key, doc, count):
        if not self.index.has_key(key):
            self.index[key] = []
        self.index[key].append((doc, count))

    def save(self, file_name):
        """
        Save the index to a file
        """
        pickle.dump(self.index, open(file_name, 'wb'))
    
    def load(self, file_name):
        self.index = pickle.load(open(file_name, 'rb'))

    def can_load(self, file_name):
        return os.path.isfile(file_name)

