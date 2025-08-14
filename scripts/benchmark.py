#!/usr/bin/env python3
import time
import tracemalloc
from pathlib import Path
from multiprocessing import cpu_count

from build_index_multi import build_index_multi
from build_index_single import build_index_single

DATA_DIR = "~/data/wikipedia10k"

def benchmark(func, label, file_paths):
    print(f"\n--- {label} ---")
    tracemalloc.start()
    start_time = time.time()

    final_dictionary, final_corpus = func(file_paths)

    elapsed_time = time.time() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    num_docs = len(final_corpus) if not callable(final_corpus) else "streamed"
    num_tokens = sum(final_dictionary.cfs.values())
    vocab_size = len(final_dictionary)

    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Peak memory: {peak / (1024 * 1024):.2f} MB")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total tokens: {num_tokens}")
    print(f"Number of documents: {num_docs}")
    if isinstance(final_corpus, list):
        avg_tokens = num_tokens / num_docs
        print(f"Average tokens/document: {avg_tokens:.2f}")

    return final_dictionary, final_corpus

if __name__ == "__main__":

    data_path = Path(DATA_DIR).expanduser()
    file_paths = list(data_path.glob("*.txt"))  # Just the paths
    print(f"Data directory: {data_path}")
    print(f"CPU cores: {cpu_count()}")

    final_dictionary_multi, final_corpus_multi = benchmark(build_index_multi, "Multi-process", file_paths)
    time.sleep(50)
    print("--------------------------------")
    final_dictionary_single, final_corpus_single = benchmark(build_index_single, "Single-process", file_paths)

    if final_dictionary_multi == final_dictionary_single:
        print("The two dictionaries are the same.")
    else:
        print("The two dictionaries are different.")

    if final_corpus_multi == final_corpus_single:
        print("The two corpora are the same.")
    else:
        print("The two corpora are different.")
    print("\nBenchmark completed.")

