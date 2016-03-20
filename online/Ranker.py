from util.Tokenizer import Tokenizer
from util.Index import Index
from util import MinHeap
import math

from collections import Counter


class Ranker:
    """
    Applying ranking algorithm from query
    """
    def __init__(self, index, tokenizer):
        assert isinstance(index, Index)
        assert isinstance(tokenizer, Tokenizer)
        self.index = index
        self.tokenizer = tokenizer


    def run_query(self, query_str, max_items=-1):
        """
        -1  - infinite queue size
        Returns:
            [dict(
                doc_id = <abc123>,
                score = <123>,
                rank = <123>
            )]
        """
        tokens = list(self.tokenizer.tokenize_str(
            query_str
            ))
        min_q = MinHeap(maxsize=max_items)
        # NOTE: the queue ranks the CLOSEST item with a high
        # score. If the ranking alg requires that the HIGHLY
        # relevant items have high scores, just output the score.
        # If the ranking alg requires the that the HIGHLY relevant 
        # items have LOW scores, multiply score by -1.
        for doc in self.index.next_doc_from_tokens(tokens):
            score = self.compute_score(doc, tokens)
            min_q.put(score, doc['doc_id'])
        rel_docs = []

        # empty out the priority queue
        while(min_q.qsize() > 0):
            score, doc_id = min_q.get_min()
            rank = min_q.qsize() + 1
            rel_docs.append(dict(
                doc_id = doc_id,
                score = score,
                rank = rank,
                ))
        # does it in place
        rel_docs.reverse()
        return rel_docs
            
        

    def compute_score(self, doc, tokens, k1=1.2, b=0.75, k2=7):
        """
        Compute the document score with BM25
        different scoring functions
        """
        score = 0
        counter = Counter(tokens)
        terms = set(tokens)
        for token in terms:
            fi = doc['term_hash'][token]['term_frequency']
            qfi = counter[token] # number of times term shows up in query
            ni = doc['term_hash'][token]['count_of_docs_with_term']
            dl = doc['doc_length']
            N = doc['coll_doc_count']
            avg_dl = doc['avg_doc_length']
            K = k1*(1-b + b*dl/avg_dl)
            t1 = (k1 + 1)*fi/(K + fi)
            t2 = (k2 + 1)*qfi/(k2 + qfi)
            t3 = math.log((N-ni+0.5)/(ni+0.5))
            score += (t1*t2*t3)
        return score



