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
    # First pass: build dictionary
    partial_dict = Dictionary(tokenized_documents_iter(path_list))
    
    # Second pass: build corpus and update frequencies
    partial_corpus = []
    for tokens in tokenized_documents_iter(path_list):
        bow = partial_dict.doc2bow(tokens)
        partial_corpus.append(bow)
    
    dict_path = f"out/tmp/partial_dict_{chunk_id}.dict"
    corpus_path = f"out/tmp/partial_corpus_{chunk_id}.mm"
    
    partial_dict.save(dict_path)
    MmCorpus.serialize(corpus_path, partial_corpus)
    
    return dict_path, corpus_path
def merge_results_from_disk(paths):
    merged_dict = Dictionary()

    # First pass: merge vocabulary incrementally
    for dict_path, _ in paths:
        part_dict = Dictionary.load(dict_path)
        merged_dict.merge_with(part_dict)

    # Second pass: rebuild cfs/dfs and remap corpus
    merged_dict.cfs.clear()
    merged_dict.dfs.clear()

    def remapped_corpus_iter():
        for dict_path, corpus_path in paths:
            part_dict = Dictionary.load(dict_path)
            token_map = {
                old_id: merged_dict.token2id[token]
                for token, old_id in part_dict.token2id.items()
                if token in merged_dict.token2id
            }

            for doc in MmCorpus(corpus_path):
                seen_tokens = set()
                remapped = []
                for old_id, freq in doc:
                    if old_id in token_map:
                        new_id = token_map[old_id]
                        remapped.append((new_id, freq))
                        merged_dict.cfs[new_id] = merged_dict.cfs.get(new_id, 0) + freq
                        if new_id not in seen_tokens:
                            merged_dict.dfs[new_id] = merged_dict.dfs.get(new_id, 0) + 1
                            seen_tokens.add(new_id)
                yield remapped

    return merged_dict, remapped_corpus_iter()


def build_index_multi(path_list):
    num_workers = cpu_count()
    chunk_size = len(path_list) // num_workers or 1
    Path("out/tmp").mkdir(parents=True, exist_ok=True)
    chunks = [path_list[i:i + chunk_size] for i in range(0, len(path_list), chunk_size)]

    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        worker_results = pool.starmap(process_chunk, [(chunk, i) for i, chunk in enumerate(chunks)])

    final_dictionary, final_corpus = merge_results_from_disk(worker_results)

    elapsed = time.time() - start_time

    final_dictionary.save("out/dictionary_multi.dict")
    MmCorpus.serialize("out/corpus_multi.mm", final_corpus, progress_cnt=10000)

    print(f"[Multi-process] Processed {len(path_list)} files in {elapsed:.2f} seconds using {num_workers} workers.")
    return final_dictionary, final_corpus


if __name__ == "__main__":
    data_path = Path("~/data/wikipedia50k").expanduser()
    file_paths = list(data_path.glob("*.txt"))  # Just the paths
    build_index_multi(path_list=file_paths)
