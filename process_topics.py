from online.Ranker import Ranker
from util.Tokenizer import Tokenizer 
from run_pipeline import build_index
from util import get_options

import sys

import pandas as pd


import time

def get_numbers_from_str(string):
    """
    Get any numbers from a string
    """
    res = []
    for c in string.strip():
        if c.isdigit():
            res.append(c)
    return ''.join(res)


def get_topics(topic_file):
    """
    Programatically extract topics
    """
    topics = []
    lines =  open(topic_file, 'r').readlines()
    topic_id = ''
    for line in lines:
        # hard code
        if line.find('<num>') == 0:
            topic_id = get_numbers_from_str(line)
            print topic_id
        if line.find('<title>') == 0:
            query = line.replace('<title>', '').strip()
            print query
            topics.append(dict(
                topicID = topic_id,
                query = query,
                ))
    return topics


if __name__ == '__main__':
    options,args = get_options()
    # get the file of the submissions from process_topics.py
    output_file = sys.argv[1]

    index = build_index(options.filename, index_file = options.index)
    tokenizer = Tokenizer()

    ranker = Ranker(index, tokenizer)
    topics = get_topics('data/topics.401-450.txt')
    all_results = []
    # topics to ignore
    ignore_topic_ids = [416, 423, 437, 444, 447]
    for topic in topics:
        if not int(topic['topicID']) in ignore_topic_ids:
            start_time = time.time()
            results = ranker.run_query(topic['query'], max_items=1000)
            run_time = time.time() - start_time
            for res in results:
                all_results.append(dict(
                    topicID = topic['topicID'],
                    q0 = 0,
                    docno = res['doc_id'],
                    rank = res['rank'],
                    score = res['score'],
                    runTag = 'cjngan_run0',
                    run_time = run_time,
                    query = topic['query'],
                    ))
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    submission_file = output_file.replace('.csv', '-submission.csv')
    submission_cols = ['topicID', 'q0', 'docno', 'rank', 'score', 'runTag']
    df[submission_cols].to_csv(
            submission_file, index=False, sep=' ', header=None)
