""""Processing raw text into json, one document is a json line"""

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
    group.add_argument('--data-type', type=str, choices=['wiki', 'books', 'document', 'wudao'])
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
args.file_size = 500 * 1024 * 1024  # 500MB

def encode_file(input_args):
    index, fn = input_args
    docs = [[]]
    processed_bytes = 0
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            processed_bytes += len(line)
            line = line.strip()
            if len(line) == 0:
                if not args.file_document:
                    docs.append([])
            else:
                docs[-1].append(line)
    
    json_list = []
    with open(os.path.join(args.output_dir, '{}.json'.format(index)), 'w') as wf:
        for doc in docs:
            if len(doc) == 0:
                continue
            data = {args.json_keys: '\n'.join(doc)}
            print(json.dumps(data), file=wf)
            # json_list.append(json.dumps(data))
    print(fn)
    return len(docs), processed_bytes

def encode_books(fn):
    docs = [[]]
    processed_bytes = 0
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            processed_bytes += len(line)
            line = line.strip()
            if len(line) != 0:
                if line.startswith('#'): # the chapter split symbol
                    docs.append([])
                docs[-1].append(line)
    
    json_list = []
    for doc in docs:
        if len(doc) == 0:
            continue
        data = {args.json_keys: '\n'.join(doc)}
    return json_list, processed_bytes

def encode_wudao(input_args):
    index, fn = input_args
    with open(fn, 'rb') as f:
        json_list = json.load(f)
    base_name = os.path.basename(fn)
    with open(os.path.join(args.output_dir, base_name), 'w', encoding='utf-8') as wf:
        for data in json_list:
            print(json.dumps(data, ensure_ascii=False), file=wf)
    return len(json_list), os.stat(fn).st_size

if args.data_type == 'wiki':
    encode_fn = encode_file
    args.file_document = False
elif args.data_type == 'document':
    encode_fn = encode_file
    args.file_document = True
elif args.data_type == 'books':
    encode_fn = encode_books
    args.file_document = False
elif args.data_type == 'wudao':
    encode_fn = encode_wudao
    args.file_document = False
else:
    raise NotImplementedError(args.data_type)

def main():
    """
    Helper script to tokenize raw text using multiple processes.
    """
    startup_start = time()

    if os.path.isfile(args.input):
        data_paths = [args.input]
    else:
        data_paths = get_all_filepath(args.input)

    data_paths = list(enumerate(data_paths))
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    def get_output_h(index):
        return open(os.path.join(output_dir, '{}.json'.format(index)), 'w', encoding='utf-8')

    # output_file_id = 0
    # output_h = get_output_h(output_file_id)
    # output_size = 0

    # create process pool
    pool = Pool(args.workers)

    # map to inputs
    encoded_lines = pool.imap_unordered(encode_fn, data_paths, args.workers)

    startup_end = time()
    proc_start = time()
    total_bytes_processed = 0
    processed_items = 0
    print("Time to startup:", startup_end - startup_start)
    print(args)
    # for json_list, bytes_processed in encoded_lines:
    for n_docs, bytes_processed in encoded_lines:

        total_bytes_processed += bytes_processed
        processed_items += n_docs
        # for item in json_list:
        #     if len(item) == 0:
        #         continue
        #     print(item, file=output_h)
        #     processed_items += 1
        #     output_size += len(item) + 1
        #     if output_size >= args.file_size:
        #         output_h.close()
        #         output_file_id += 1
        #         output_h = get_output_h(output_file_id)
        #         output_size = 0
        if processed_items % args.log_interval == 0:
            current = time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {processed_items} documents",
                f"({processed_items/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr)

    # output_h.close()
    total_time = time() - startup_start
    print("total costing time: {:.3f}s".format(total_time))


if __name__ == "__main__":
    main()


