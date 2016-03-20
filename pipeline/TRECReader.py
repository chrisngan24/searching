"""
Reads the TREC latimes dataset
"""
import gzip
import re

from xml.dom import pulldom
import xml

from util.Tokenizer import Tokenizer

class TRECReader:
    def __init__(self, 
            trec_file,
            doc_id_tag='DOCNO',
            doc_end_tag='</DOC>',
            relevant_tags=[
                'HEADLINE',
                'TEXT',
                'GRAPHIC',
                'SUBJECT', # as per given src code
                ]
            ):
        if trec_file.endswith('.gz'):
            # Reads gzip files
            self.file_stream = gzip.open(trec_file, 'r')
        else:
            # general text file
            self.file_stream = open(trec_file, 'r')
        self.infile_name = trec_file
        self.doc_id_tag = doc_id_tag
        self.doc_end_tag = doc_end_tag 
        self.relevant_tags = relevant_tags + [doc_id_tag]


    def next_doc(self):
        """
        Load one doc at a time, no need to load whole thing in mem
        Reads a single doc into memory so that it
        can be processed by an xml tree
        """
        doc_content = ''
        for line in self.file_stream:
            line = line.strip() + ' '
            
            doc_content += line
            if line.find(self.doc_end_tag) == 0:
                yield doc_content
                doc_content = ''
                



    def stream_docs(self):
        """
        Generator that streams a dictionary
        with:
          @doc_id      - identifier of the document
          @doc_content - raw text of the document
        """
        ## Pull out the doc
        #for event, node in self.pull_xml_parser():
        for doc_content in  self.next_doc():
            tag_stack = []
            # parse the xml into a streamer
            str_buffer = []
            doc_id = ''
            for event, node in pulldom.parseString(doc_content):
                if node.nodeName in self.relevant_tags:
                    if event == pulldom.START_ELEMENT:
                        # next set of text is within
                        # some relevant tag
                        tag_stack.append(node.nodeName)
                    if event == pulldom.END_ELEMENT:
                        tag_stack.pop()
                elif event == pulldom.CHARACTERS and len(tag_stack) > 0:
                    # is within a relevant stack an
                    # interested in reading the document

                    peeked_val = tag_stack[len(tag_stack)-1]
                    if peeked_val == self.doc_id_tag:
                        # This is the docID
                        doc_id = node.nodeValue.strip()
                    else:
                        str_buffer.append(node.nodeValue)

            # so we don't have to keep creating new strings
            doc_content = ''.join(str_buffer)
            yield dict(
                    doc_id=doc_id,
                    doc_content=doc_content,
                    )
