import pickle
import os
import sys

import jsonpickle 
import json

from util.Lexicon import Lexicon

class Index:
    def __init__(self):
        self.index = {}
        self.term_lexicon = Lexicon()
        self.doc_lexicon = Lexicon()
        # tracks the length of each documnet
        self.doc_length_count = {}
        # tracks the number of occurences per term
        self.term_length_count = {}
        # number of words in the collection
        self.coll_term_count = 0.
        # number of docs in the collection
        self.coll_doc_count = 0.
        # aerage document length
        self.avg_doc_length = 0.


    def get_doc_key(self, doc):
        dv = self.doc_lexicon.map_k_to_v(doc)

        if not self.doc_length_count.has_key(dv):
            # a new document
            self.doc_length_count[dv] = 0
            self.coll_doc_count += 1.
        return dv

    def get_term_key(self, term):
        tk = self.term_lexicon.map_k_to_v(term)
        if not self.term_length_count.has_key(tk):
            # a new term
            self.term_length_count[tk] = 0
        if not self.index.has_key(tk):
            # a new term
            self.index[tk] = []
        return tk


    def put(self,term, doc, count):
        tk = self.get_term_key(term)
        dv = self.get_doc_key(doc)
        self.doc_length_count[dv] += count 
        self.coll_term_count += count
        self.term_length_count[tk] += count

        self.index[tk].append(dv)
        self.index[tk].append(count)

        self.avg_doc_length = self.coll_term_count / self.coll_doc_count


    def _get_term_doc_list(self, term):
        tk = self.term_lexicon.map_k_to_v(term)
        return self.index[tk]

    def get_term_frequency_dict(self, term):
        doc_list = self._get_term_doc_list(term)
        # assert its even
        assert(doc_list % 2 == 0)

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

    def next_doc_from_tokens(self, tokens):
        """
        Applies doc at a time query processing
        """

        # remove tokens that are not part of the vocab
        rel_tokens = []
        for t in tokens:
            if self.term_lexicon.has_key(t):
                rel_tokens.append(t)
            else:
                print t, ': term does not exist in vocab'
        pointers = [0]*len(rel_tokens)
        has_docs = True
        # the list of doc/counts per term
        rel_term_list = map(
                self._get_term_doc_list,
                rel_tokens
                )
        n = len(rel_tokens)
        while(has_docs):
            # assume that the max int will never be reached
            next_doc_id = sys.maxint
            for i in xrange(n):
                term_list_pointer = pointers[i]
                if term_list_pointer != -1:
                    d_id = rel_term_list[i][term_list_pointer]
                    # assumes that document ids
                    # are stored in incremental order
                    next_doc_id = min(d_id, next_doc_id)
            if next_doc_id == sys.maxint:
                print 'Done query processing'
                # processed all docs, stop while loop
                has_docs = False
            else:
                # relevant docs still exist


                # create the hash that will be returned for
                # the ranker. Just dump allll the data
                doc_data = dict(
                        doc_id = self.doc_lexicon.map_v_to_k(next_doc_id),
                        # number of terms in document
                        coll_term_count = self.coll_term_count,
                        # number of document in collection 
                        coll_doc_count = self.coll_doc_count,
                        # assumes all term freq are init zero
                        term_hash= { term : dict(
                            count_of_docs_with_term=0,
                            term_frequency = 0,
                            ) \
                                for term in tokens },
                        doc_length = self.doc_length_count[next_doc_id],
                        avg_doc_length = self.avg_doc_length,
                        )
                for i in xrange(n):
                    term = rel_tokens[i]
                    t_li_pointer = pointers[i]
                    term_li = rel_term_list[i]
                    # only iterate if the term list has any more documents
                    # to process. Only process the items that have the
                    # document id that is being processed
                    if t_li_pointer != -1 and term_li[t_li_pointer] == next_doc_id:
                        if len(term_li) <= (t_li_pointer + 2):
                            # this documet is the last one
                            # that has this term. Stop 
                            # iterating after this
                            pointers[i] = -1
                        else:
                            # the list is a pairing of 
                            # doc IDs and the term count
                            pointers[i] += 2

                        count = term_li[t_li_pointer + 1]
                        # update the doc raw data
                        doc_data['term_hash'][term]['term_frequency'] = float(count)
                        doc_data['term_hash'][term]['count_of_docs_with_term'] = \
                                len(term_li)/2.
                                     
                yield doc_data

                



    
    def load(self, file_name):
        """
        Load index from a file
        """
        return pickle.load(open(file_name, 'rb'))


    def can_load(self, file_name):
        return os.path.isfile(file_name)

