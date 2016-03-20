
import os


def append_to_file(infile_path, outfile):
    with open(infile_path, 'r') as infile:
        for line in infile:
            outfile.write(line)

def write_files(f, outfile):
    if os.path.isfile(f) and f.endswith('.py'):
        outfile.write('## %s\n\n' % f)
        outfile.write('~~~{.py}\n')
        append_to_file(f, outfile)
        outfile.write('~~~\n\n')
    if os.path.isdir(f):
        outfile.write('# %s \n\n' % f)
        for x in os.listdir(f):
            write_files(
                    '%s/%s' % (f,x),
                    outfile,
                    )
         

if __name__ == '__main__':
    files_and_folders = [
            'run_pipeline.py',
            'run_online.py',
            'process_topics.py',
            'evaluate_results.py',
            'pipeline',
            'util',
            'online',
            'evaluate',
            ]
    output_file = 'report.md'
    with open(output_file, 'w') as outfile:
        outfile.write('''
---
title:  'Search Engine - Project Part 1 Source Code'
author: Christopher Ngan - 20423692
---\n\n


# Introduction

Report is structured to match the directory structure of the code. Base directory represents the home directory that all scripts are ran from.

```

run_pipeline.py
run_offline.py
process_topics.py
evaluate_results.py
    util/...
    online/..
    pipeline/..
    evaluate/..

```

To build the index/run the pipeline, run:

```

python run_pipeline.py --index <path to save index to/load data from> --file <path to LAtimes dataset>

```

To run the search engine online, run:

```

python run_online.py --index <path to save index to/load data from> --file <path to LAtimes dataset>

```

        ''')
        outfile.write('\n# Base Directory\n\n')
        for f in files_and_folders:
            write_files(f, outfile)

