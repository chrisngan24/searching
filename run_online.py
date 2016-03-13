from online.Ranker import Ranker
from util.Tokenizer import Tokenizer 
from run_pipeline import build_index

from util import get_options


if __name__ == "__main__":
    options,args = get_options()

    index = build_index(options.filename, index_file = options.index)
    tokenizer = Tokenizer()

    ranker = Ranker(index, tokenizer)
    print 'Search Engine is online!'
    while (True):
        query = raw_input('===Please enter query===\n')
        print ranker.run_query(query, max_items=1000)


