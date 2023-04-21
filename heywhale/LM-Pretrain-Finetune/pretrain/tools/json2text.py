""""Processing json into text one document is a json line"""

import contextlib
import os
import sys
from collections import Counter
from multiprocessing import Pool
from itertools import chain
from time import time
import argparse
import json
import gzip
import nltk

def get_all_filepath(dir_path):
    files = []
    for dn, dp, fp in os.walk(dir_path):
        for f in fp:
            files.append(os.path.join(dn, f))
    return list(sorted(files))

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input text')
    group.add_argument('--json-keys', type=str, default='text',
                       help='space separate listed of keys to extract from json')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-dir', type=str, required=True,
                       help='Path to output dir')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    args = parser.parse_args()

    return args

args = get_args()
args.file_size = 10 * 1024 * 1024  # 10MB

class Encoder(object):
    def initializer(self):
        
        Encoder.splitter = nltk.load("tokenizers/punkt/english.pickle")

    def encode_fn(self, line):
        processed_bytes = len(line)
        line = line.strip()
        if len(line) == 0:
            return "", processed_bytes
        data = json.loads(line)
        text = data[args.json_keys]
        doc = []
        for sent in text.split('\n'):
            for sent1 in Encoder.splitter.tokenize(sent):
                if len(sent1) > 0:
                    doc.append(sent1)
        res = ""
        if len(doc) > 0:
            res = "\n".join(doc)
        return res, processed_bytes

def main():
    """
    Helper script to tokenize raw text using multiple processes.
    """
    startup_start = time()

    if os.path.isfile(args.input):
        data_paths = [args.input]
    else:
        data_paths = get_all_filepath(args.input)


    inputs = [gzip.open(input, 'rb') for input in data_paths]
    fin = chain(*inputs)

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    def get_output_h(index):
        return open(os.path.join(output_dir, '{}.txt'.format(index)), 'w', encoding='utf-8')

    output_file_id = 0
    output_h = get_output_h(output_file_id)
    output_size = 0

    encoder = Encoder()
    # create process pool
    pool = Pool(args.workers, initializer=encoder.initializer)

    # map to inputs
    encoded_lines = pool.imap(encoder.encode_fn, fin, args.workers * 10)

    startup_end = time()
    proc_start = time()
    total_bytes_processed = 0
    processed_items = 0
    print("Time to startup:", startup_end - startup_start)
    print(args)
    processed_items = 0
    for item, bytes_processed in encoded_lines:
        total_bytes_processed += bytes_processed
        if len(item) == 0:
            continue
        print(item, file=output_h)
        print("", file=output_h)
        processed_items += 1
        output_size += len(item) + 1
        if output_size >= args.file_size:
            output_h.close()
            output_file_id += 1
            output_h = get_output_h(output_file_id)
            output_size = 0
        if processed_items % args.log_interval == 0:
            current = time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {processed_items} documents",
                f"({processed_items/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr)

    output_h.close()
    total_time = time() - startup_start
    print("total costing time: {:.3f}s".format(total_time))


if __name__ == "__main__":
    main()


