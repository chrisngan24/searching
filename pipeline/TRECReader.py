"""
Reads the TREC latimes dataset
"""
import gzip
import re

from xml.dom import pulldom
import xml.etree.ElementTree as ET
import xml
class TRECReader:
    def __init__(self, 
            trec_file,
            doc_id_tag='DOCNO',
            doc_end_tag='</DOC>',
            relevant_tags=[
                'HEADLINE',
                'TEXT',
                'GRAPHIC',
                ]
            ):
        # Reads gzip files
            # general text file
        self.infile_name = trec_file
        self.doc_id_tag = doc_id_tag
        self.doc_end_tag = doc_end_tag 
        self.file_stream = open(self.infile_name, 'r')
        self.doc_streamer = pulldom.parse(self.file_stream)
        self.relevant_tags = relevant_tags + [doc_id_tag]


    def next_doc(self):
        """
        Load one doc at a time, no need to load whole thing in mem
        """
        doc_content = ''
        for line in self.file_stream:
            line = line.strip() + ' '
            
            doc_content += line
            if line.find(self.doc_end_tag) == 0:
                yield doc_content
                doc_content = ''
                



    def stream_docs(self):
        ## Pull out the doc
        #for event, node in self.pull_xml_parser():
        for doc_content in  self.next_doc():
            tag_stack = []
            # parse the xml into a streamer
            for event, node in pulldom.parseString(doc_content):
                if node.nodeName in self.relevant_tags:
                    if event == pulldom.START_ELEMENT:
                        tag_stack.append(node.nodeName)
                    if event == pulldom.END_ELEMENT:
                        tag_stack.pop()
                elif event == pulldom.CHARACTERS and len(tag_stack) > 0:
                    # peek the stack
                    peeked_val = tag_stack[len(tag_stack)-1]
                    if peeked_val == self.doc_id_tag:
                        doc_id = node.nodeValue
                        #print doc_id
                    else:
                        print node.nodeValue
                        pass
