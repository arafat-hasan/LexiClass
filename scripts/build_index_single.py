#!/usr/bin/env python3
import time
from pathlib import Path
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary, MmCorpus

def build_index_single(file_paths):
    tokenized_docs = [simple_preprocess(Path(fp).read_text()) for fp in file_paths]
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    return dictionary, corpus

if __name__ == "__main__":
    all_files = list(Path("~/data/wikipedia100k").expanduser().glob("*.txt"))

    start_time = time.time()
    dictionary, corpus = build_index_single(all_files)
    elapsed = time.time() - start_time

    dictionary.save("dictionary_single.dict")
    MmCorpus.serialize("corpus_single.mm", corpus)

    print(f"[Single-threaded] Processed {len(all_files)} documents in {elapsed:.2f} seconds.")
