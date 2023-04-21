
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BasicTokenizer:
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=False, split_on_punc=True):
        """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
        self.do_lower_case = do_lower_case
        self.split_on_punc = split_on_punc
        self.white_space_tok = Whitespace()

    def tokenize(self, i, norm_text):
        """Tokenizes a piece of text."""
        text = str(norm_text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            if self.split_on_punc:
                split_tokens.extend(self._run_split_on_punc(token))
            else:
                split_tokens.append(token)

        s =  NormalizedString(" ".join(split_tokens))
        return s.split(' ', behavior='removed')
    
    def pre_tokenize(self, pretok):
        pretok.split(self.tokenize)

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


import re
class CustomDecoder:
    def decode(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        out_string = re.sub(' [^a-zA-Z0-9]', '', out_string)
        return out_string

from tokenizers import Tokenizer, NormalizedString
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import WordPieceTrainer, BpeTrainer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents


from tokenizers.pre_tokenizers import Whitespace, PreTokenizer
from tokenizers.decoders import Decoder


def get_all_filepath(dir_path):
    files = []
    for dn, dp, fp in os.walk(dir_path):
        for f in fp:
            files.append(os.path.join(dn, f))
    return list(sorted(files))

def train_tokenizer(data_dir, output_path):
    trainer = WordPieceTrainer(
        vocab_size=51280,
        min_frequency=10,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
        continuing_subword_prefix='##'
    )
    tokenizer = initialize_tokenizer()
    files = get_all_filepath(data_dir)
    tokenizer.train(files, trainer)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(output_path)

def initialize_tokenizer(path=None):
    if path is None:
        tokenizer = Tokenizer(WordPiece(unk_token='<unk>'))
        normalizer = normalizers.Sequence([NFD(), StripAccents()])
        tokenizer.normalizer = normalizer
    else:
        tokenizer = Tokenizer.from_file(path)
        
    basic_tokenizer = BasicTokenizer()
    tokenizer.pre_tokenizer = PreTokenizer.custom(basic_tokenizer)
    return tokenizer


if __name__ == '__main__':
    import os, sys
    data_dir = sys.argv[1]
    output_path = sys.argv[2]
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    train_tokenizer(data_dir, output_path)
