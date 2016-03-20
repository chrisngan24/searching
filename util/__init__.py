"""
Util classes used in both
"""
from optparse import OptionParser

import heapq

def get_options():
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
            help="Data file to load data from",
            )
    parser.add_option("-i", "--index", dest="index",
            help="File to load/save index from",
            default='',
            )
    options, args = parser.parse_args()
    if not options.filename:   # if filename is not given
        parser.error('Filename not given')

    return (options, args)




class MinHeap:
    def __init__(self, maxsize=-1):
        self.heap = []
        self.maxsize = maxsize

    def qsize(self):
        return len(self.heap)

    def put(self, score, data):
        heapq.heappush(self.heap, (score,data))
        if self.maxsize != -1 and self.qsize() > self.maxsize:
            heapq.heappop(self.heap)
    def get_min(self):
        return heapq.heappop(self.heap)
