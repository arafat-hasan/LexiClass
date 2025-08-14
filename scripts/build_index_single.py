#!/usr/bin/env python3
import time
from pathlib import Path
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary, MmCorpus

def iter_documents(path_list):
    for file_path in path_list:
        fp = Path(file_path).expanduser()
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            yield f.read()
            
def tokenized_documents_iter(path_list):
    for doc in iter_documents(path_list):
        yield simple_preprocess(doc)

def process_full(path_list):
    full_dictionary = Dictionary(tokenized_documents_iter(path_list))
    full_corpus = [full_dictionary.doc2bow(tokens) for tokens in tokenized_documents_iter(path_list)]
    return full_dictionary, full_corpus

def build_index_single(file_paths):
    start_time = time.time()

    final_dictionary, final_corpus = process_full(file_paths)

    elapsed = time.time() - start_time

    final_dictionary.save("dictionary_single.dict")
    MmCorpus.serialize("corpus_single.mm", final_corpus)

    print(f"[Single-process] Processed {len(file_paths)} files in {elapsed:.2f} seconds.")
    return final_dictionary, final_corpus

if __name__ == "__main__":
    data_path = Path("~/data/wikipedia50k").expanduser()
    file_paths = list(data_path.glob("*.txt"))  # Just the paths
    build_index_single(file_paths)
