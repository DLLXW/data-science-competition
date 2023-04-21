# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import re

import torch
try:
    import nltk
    nltk_available = False
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from itertools import chain
import contextlib
from transformers import BartTokenizer, BertTokenizer

def chain_files(inputs):
    new_lined_inputs = []
    for input in inputs:
        new_lined_inputs.append(input)
        new_lined_inputs.append("\n")
    return chain(*new_lined_inputs)

def get_all_filepath(dir_path):
    files = []
    for dn, dp, fp in os.walk(dir_path):
        for f in fp:
            files.append(os.path.join(dn, f))
    return list(sorted(files))


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.data_paths = args.data_paths

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        # Encoder.tokenizer = BertTokenizer.from_pretrained(self.args.vocab_file)
        Encoder.tokenizer.model_max_length = 999999 # avoid warnings
        stop_sent = '?!。？！;；,，'
        Encoder.stop_pattern = re.compile('[' + stop_sent + ']')
        if False and self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def split_sent(self, sent):
        min_len = 0
        sent_list = []
        start = 0
        for match in Encoder.stop_pattern.finditer(sent.strip()):
            end = match.end()
            sent_len = end - start
            if sent_len < min_len:
                continue
            else:
                sent_list.append(sent[start:end])
                start = end
        if len(sent) - start > min_len:
            sent_list.append(sent[start:])
        return sent_list

    def cut_sent(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{3})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{1})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    def encode_json(self, json_line):
        if len(json_line.strip()) == 0:
            return {}, len(json_line)
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for line in text.split('\n'):
                # for sentence in Encoder.splitter.tokenize(line):
                for sentence in self.cut_sent(line):
                    #sentence_ids = Encoder.tokenizer.tokenize(sentence)
                    # sentence_ids = Encoder.tokenizer.convert_tokens_to_ids(sentence_ids)
                    sentence_ids = [int(i)+100 for i in sentence.split(' ')]
                    if len(sentence_ids) > 0:
                        doc_ids.append(sentence_ids)
                if len(doc_ids) > 0 and self.args.append_eod:
                    doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(json_line)

    def encode_file(self, fn_i):
        docs = []
        fn = self.data_paths[fn_i]
        with open(fn, 'r', encoding='utf-8') as f:
            doc_ids = []
            for line in f:
                line = line.strip()
                if len(line) == 0 and len(doc_ids) > 0:
                    docs.append(doc_ids)
                    doc_ids = []
                for sentence in self.cut_sent(line):
                    for sentence in Encoder.splitter.tokenize(sentence):
                        sentence_ids = Encoder.tokenizer.tokenize(sentence)
                        sentence_ids = Encoder.tokenizer.convert_tokens_to_ids(sentence_ids)
                        if len(sentence_ids) > 0:
                            doc_ids.append(sentence_ids)
        if len(doc_ids):
            docs.append(doc_ids)
        docs = [d for d in docs if len(d) > 0]
        print(fn_i, fn, os.stat(fn).st_size, flush=True)
        return docs, os.stat(fn).st_size
            
    
    def encode(self, text_line):
        assert len(self.args.json_keys) == 1
        doc_ids = []
        for sentence in Encoder.splitter.tokenize(text_line):
            sentence_ids = Encoder.tokenizer.tokenize(sentence)
            if len(sentence_ids) > 0:
                doc_ids.append(sentence_ids)
        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids[-1].append(Encoder.tokenizer.eod)
        ids = {self.args.json_keys[0]: doc_ids}
        return ids, len(text_line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--use-gzip', action='store_true', default=False,
                       help='use gzip to open file')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'Huggingface'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')


    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=10000,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False


    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

START_ID = 0
def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    if os.path.isfile(args.input):
        data_paths = [args.input]
    else:
        data_paths = get_all_filepath(args.input)
    with contextlib.ExitStack() as stack:
        # open files
        inputs = [
            stack.enter_context(gzip.open(input, 'rt', encoding='utf-8') if args.use_gzip else open(input, 'r'))
            if input != "-"
            else sys.stdin
            for input in data_paths
        ]
        # # fin = open(args.input, 'r', encoding='utf-8')
        fin = chain(*inputs)
        # fin = list(range(START_ID, len(data_paths)))
        args.data_paths = data_paths

        # if nltk_available and args.split_sentences:
            # print('Preparing nltk.')
            # nltk.download("punkt", quiet=True)

        encoder = Encoder(args)
        tokenizer = build_tokenizer(args)
        # tokenizer = BertTokenizer.from_pretrained(args.vocab_file)
        tokenizer.model_max_length = 999999 # avoid warnings

        if str(tokenizer.__class__).lower().startswith('bert'):
            if not args.split_sentences:
                print("Bert tokenizer detected, are you sure you don't want to split sentences?")

        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode_json, fin, 25)
        # encoded_files = pool.imap_unordered(encoder.encode_file, fin, 25)

        level = "document"
        if args.split_sentences:
            level = "sentence"

        print(f"Vocab size: {tokenizer.vocab_size}", flush=True)
        print(f"Output prefix: {args.output_prefix}", flush=True)
        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        cur_size = {}
        shard_size = 1024*1024*1024*10/4  # 10 GB, int type
        shard_id = {}

        def get_new_builder(key_name, i):
            output_bin_files[key] = "{}_{}_{}_{}.bin".format(args.output_prefix,
                                                        key, level, i)
            output_idx_files[key] = "{}_{}_{}_{}.idx".format(args.output_prefix,
                                                        key, level, i)
            builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                impl=args.dataset_impl,
                                                vocab_size=tokenizer.vocab_size)
            
        for key in args.json_keys:
            cur_size[key] = 0
            shard_id[key] = 0
            get_new_builder(key, shard_id[key])

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start, flush=True)

        for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key, sentences in doc.items():
                if len(sentences) == 0:
                    continue
                for sentence in sentences:
                    item = torch.IntTensor(sentence)
                    builders[key].add_item(item)
                    cur_size[key] += item.shape[0]
                builders[key].end_document()

            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed/elapsed/1024/1024
                print(f"Processed {i} documents",
                    f"({i/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)
            
            for key in args.json_keys:
                if cur_size[key] > shard_size:
                    builders[key].finalize(output_idx_files[key])
                    cur_size[key] = 0
                    shard_id[key] += 1
                    get_new_builder(key, shard_id[key])
                    print("sharding dataset key:{}, shard_id:{}".format(key, shard_id[key]))

        # n_docs = 0
        # for i, (docs, bytes_processed) in enumerate(encoded_files, start=1):
        #     total_bytes_processed += bytes_processed
        #     for sentences in docs:
        #         if len(sentences) == 0:
        #             continue
        #         for sentence in sentences:
        #             item = torch.tensor(sentence, dtype=torch.int32)
        #             builders[key].add_item(item)
        #             cur_size[key] += item.shape[0]
        #         builders[key].end_document()
        #         n_docs += 1
        #         if n_docs % args.log_interval == 0:
        #             current = time.time()
        #             elapsed = current - proc_start
        #             mbs = total_bytes_processed/elapsed/1024/1024
        #             print(f"Processed {n_docs} documents",
        #                 f"({i/elapsed} docs/s, {mbs} MB/s).",
        #                 file=sys.stderr, flush=True)
                
        #         for key in args.json_keys:
        #             if cur_size[key] > shard_size:
        #                 builders[key].finalize(output_idx_files[key])
        #                 cur_size[key] = 0
        #                 shard_id[key] += 1
        #                 get_new_builder(key, shard_id[key])
        #                 print("sharding dataset key:{}, shard_id:{}".format(key, shard_id[key]), flush=True)


        for key in args.json_keys:
            builders[key].finalize(output_idx_files[key])

if __name__ == '__main__':
    main()
