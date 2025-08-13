#!/usr/bin/env python3
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary, MmCorpus

def process_batch(file_paths):
    tokenized_docs = [simple_preprocess(Path(fp).read_text()) for fp in file_paths]
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    return dictionary, corpus

def merge_results(results):
    merged_dict = Dictionary()
    merged_corpus = []
    for dictionary, corpus in results:
        merged_dict.merge_with(dictionary)
        merged_corpus.extend(corpus)
    return merged_dict, merged_corpus

def build_index_multi(file_paths, batch_size=1000):
    batches = [file_paths[i:i+batch_size] for i in range(0, len(file_paths), batch_size)]
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_batch, batches)
    return merge_results(results)

if __name__ == "__main__":
    all_files = list(Path("~/data/wikipedia100k").expanduser().glob("*.txt"))

    start_time = time.time()
    dictionary, corpus = build_index_multi(all_files, batch_size=100)
    elapsed = time.time() - start_time

    dictionary.save("dictionary_multi.dict")
    MmCorpus.serialize("corpus_multi.mm", corpus)

    print(f"[Multiprocessing] Processed {len(all_files)} documents in {elapsed:.2f} seconds using {cpu_count()} cores.")
