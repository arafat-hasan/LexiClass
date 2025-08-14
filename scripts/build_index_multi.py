#!/usr/bin/env python3
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
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


def process_chunk(path_list, chunk_id):
    partial_dict = Dictionary(tokenized_documents_iter(path_list))
    partial_corpus = (partial_dict.doc2bow(tokens) for tokens in tokenized_documents_iter(path_list))

    dict_path = f"tmp/partial_dict_{chunk_id}.dict"
    corpus_path = f"tmp/partial_corpus_{chunk_id}.mm"

    partial_dict.save(dict_path)
    MmCorpus.serialize(corpus_path, partial_corpus)

    return dict_path, corpus_path

# def merge_results_from_disk(paths):
#     merged_dict = Dictionary.load(paths[0][0])

#     for dict_path, _ in paths[1:]:
#         part_dict = Dictionary.load(dict_path)
#         merged_dict.merge_with(part_dict)

#     merged_corpus = []
#     for dict_path, corpus_path in paths:
#         part_dict = Dictionary.load(dict_path)
#         id_map = {old_id: merged_dict.token2id[token]
#                   for token, old_id in part_dict.token2id.items()
#                   if token in merged_dict.token2id}

#         for doc in MmCorpus(corpus_path):
#             remapped_doc = [(id_map[word_id], freq) for word_id, freq in doc if word_id in id_map]
#             merged_corpus.append(remapped_doc)

#     return merged_dict, merged_corpus

def merge_results_from_disk(paths):
    merged_dict = Dictionary.load(paths[0][0])
    for dict_path, _ in paths[1:]:
        part_dict = Dictionary.load(dict_path)
        merged_dict.merge_with(part_dict)

    def remapped_corpus_iter():
        for dict_path, corpus_path in paths:
            part_dict = Dictionary.load(dict_path)
            id_map = {old_id: merged_dict.token2id[token]
                      for token, old_id in part_dict.token2id.items()
                      if token in merged_dict.token2id}

            for doc in MmCorpus(corpus_path):
                yield [(id_map[word_id], freq) for word_id, freq in doc if word_id in id_map]

    return merged_dict, remapped_corpus_iter()

def build_index_multi(file_paths):
    num_workers = cpu_count()
    chunk_size = len(file_paths) // num_workers or 1
    Path("tmp").mkdir(exist_ok=True)
    chunks = [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]

    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        worker_results = pool.starmap(process_chunk, [(chunk, i) for i, chunk in enumerate(chunks)])

    final_dictionary, final_corpus = merge_results_from_disk(worker_results)

    elapsed = time.time() - start_time

    final_dictionary.save("dictionary_multi.dict")
    MmCorpus.serialize("corpus_multi.mm", final_corpus)

    print(f"[Multi-process] Processed {len(file_paths)} files in {elapsed:.2f} seconds using {num_workers} workers.")
    return final_dictionary, final_corpus


if __name__ == "__main__":
    data_path = Path("~/data/wikipedia50k").expanduser()
    file_paths = list(data_path.glob("*.txt"))  # Just the paths
    build_index_multi(file_paths)
