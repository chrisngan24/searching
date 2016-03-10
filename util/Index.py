import pickle
import os

import jsonpickle 
import json

from util.Lexicon import Lexicon

class Index:
    def __init__(self):
        self.index = {}
        self.term_lexicon = Lexicon()
        self.doc_lexicon = Lexicon()

    def put(self,term, doc, count):
        tk = self.term_lexicon.map_k_to_v(term)
        dv = self.doc_lexicon.map_k_to_v(doc)
        if not self.index.has_key(tk):
            self.index[tk] = []
        #self.index[tk].append((dv, count))
        self.index[tk].append(dv)
        self.index[tk].append(count)

    def get_term(self, term):
        tk = self.term_lexicon.map_k_to_v(term)
        return self.index[tk]

    def save(self, file_name):
        """
        Save the index to a file
        """
        pickle.dump(self, open(file_name, 'wb'))
        '''
        Can improve RAM usage, but NBD
        This likely decreases performance as you 
        have to load a string into memory
        '''
        '''
        encoded = jsonpickle.encode(self)
        with open(file_name, 'w') as outfile:
            json.dump(encoded, outfile)
        '''
    
    def load(self, file_name):
        """
        Load index from a file
        """
        return pickle.load(open(file_name, 'rb'))


    def can_load(self, file_name):
        return os.path.isfile(file_name)

