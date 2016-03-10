from pipeline.TRECReader import TRECReader

if __name__ == '__main__':
    tr = TRECReader('data/latimes.sample.txt')
    #tr = TRECReader('data/latimes')
    tr.stream_docs()
