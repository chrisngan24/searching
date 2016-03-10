from pipeline.TRECReader import TRECReader
from util.Tokenizer import Tokenizer
from util.Index import Index 

from optparse import OptionParser

def get_options():
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
            help="Data file to load data from",
            default='data/latimes.sample.txt')
    parser.add_option("-i", "--index",
            help="File to load/save index from",
            default='',
            )

    return parser.parse_args()



if __name__ == '__main__':

    option, args = get_options()

    index_file = option.index 

    index = Index() 
    if index_file == '' or \
        (index_file != '' and not index.can_load(index_file)):
        print 'Recomputing the index'
        tr = TRECReader(option.filename)
        t = Tokenizer()
        doc_count = 0
        
        for doc in tr.stream_docs():
            doc_id = doc['doc_id']
            doc_content = doc['doc_content']
            print 'Processing document', doc_id
            # dictionary to count term occurrences in the document
            term_counter = {}
            for token in t.tokenize_str(doc_content):
                if not term_counter.has_key(token):
                    term_counter[token] = 0
                term_counter[token] += 1
            for k in term_counter.keys():
                index.put(k, doc_id, term_counter[k])
            doc_count += 1
        print doc_count, 'documents processed'
        if index_file != '' and not index.can_load(index_file):
            print 'saving the index to', index_file
            index.save(index_file)
    else:
        print 'Loading the index from', index_file
        index = index.load(index_file)
    import pdb; pdb.set_trace()
