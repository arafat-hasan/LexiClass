#!/usr/bin/env python3
import time
import tracemalloc
from pathlib import Path
from multiprocessing import cpu_count
from gensim.corpora import MmCorpus
from build_index_multi import build_index_multi
from build_index_single import build_index_single

DATA_DIR = Path("~/data/wikipedia1k").expanduser()
    
# Compare entire corpora while maintaining memory efficiency
def compare_corpora():
    # Load the serialized corpora
    corpus_multi_mm = MmCorpus("out/corpus_multi.mm")
    corpus_single_mm = MmCorpus("out/corpus_single.mm")
    
    def get_iterator(corpus):
        return iter(corpus)
    
    iter_multi = get_iterator(corpus_multi_mm)
    iter_single = get_iterator(corpus_single_mm)
    
    docs_compared = 0
    mismatches = 0
    
    try:
        while True:
            try:
                doc1 = next(iter_multi)
                doc2 = next(iter_single)
                
                if doc1 != doc2:
                    mismatches += 1
                    print(f"Mismatch at document {docs_compared}")
                
                docs_compared += 1
                
                if docs_compared % 1000 == 0:
                    print(f"Compared {docs_compared} documents...")
                    
            except StopIteration:
                break
                
        # Check if one iterator is exhausted before the other
        try:
            next(iter_multi)
            print("Error: Multi-process corpus has more documents")
            return False
        except StopIteration:
            try:
                next(iter_single)
                print("Error: Single-process corpus has more documents")
                return False
            except StopIteration:
                pass
                
        print("\nCorpus comparison completed:")
        print(f"Total documents compared: {docs_compared}")
        print(f"Total mismatches found: {mismatches}")
        return mismatches == 0
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        return False


def benchmark(func, label, path_list):
    print(f"\n--- {label} ---")
    tracemalloc.start()
    start_time = time.time()

    final_dictionary, final_corpus = func(path_list)

    elapsed_time = time.time() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    vocab_size = len(final_dictionary)
    num_tokens = sum(final_dictionary.cfs.values())
    

    is_generator = callable(final_corpus) or hasattr(final_corpus, '__iter__') and not hasattr(final_corpus, '__len__')
    
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Peak memory: {peak / (1024 * 1024):.2f} MB")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total tokens: {num_tokens}")
    print(f"Documents: {'streamed' if is_generator else len(final_corpus)}")
    
    if not is_generator:
        avg_tokens = num_tokens / len(final_corpus)
        print(f"Average tokens/document: {avg_tokens:.2f}")

    return final_dictionary, final_corpus

if __name__ == "__main__":

    data_path = Path(DATA_DIR).expanduser()
    file_paths = list(data_path.glob("*.txt"))  # Just the paths
    print(f"Data directory: {data_path}")
    print(f"CPU cores: {cpu_count()}")

    final_dictionary_multi, final_corpus_multi = benchmark(build_index_multi, "Multi-process", file_paths)
    # Allow system to stabilize
    time.sleep(2)
    print("--------------------------------")
    final_dictionary_single, final_corpus_single = benchmark(build_index_single, "Single-process", file_paths)

    # Compare dictionaries
    if final_dictionary_multi == final_dictionary_single:
        print("The two dictionaries are the same.")
    else:
        print("The two dictionaries are different.")
    
    print("\nComparing entire corpora...")
    if compare_corpora():
        print("The corpora are identical!")
    else:
        print("The corpora differ.")
    print("\nBenchmark completed.")
