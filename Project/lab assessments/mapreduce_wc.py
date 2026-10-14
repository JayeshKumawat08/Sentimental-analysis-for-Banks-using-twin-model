import argparse
import multiprocessing as mp
import re
import time
import zlib
from collections import Counter

# Regular expression used to extract words
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def read_chunks(file_path, lines_per_chunk):
    chunk = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            chunk.append(line)
            if len(chunk) == lines_per_chunk:
                yield chunk
                chunk = []
    if chunk:
        yield chunk

def mapper(lines):
    local_counts = Counter()
    for line in lines:
        words = TOKEN_RE.findall(line)
        for word in words:
            local_counts[word.lower()] += 1
    return local_counts

def map_reduce_word_count(file_path, workers=2, reducers=4, lines_per_chunk=10000):
    partitions = [Counter() for _ in range(reducers)]
    with mp.Pool(processes=workers) as pool:
        chunks = read_chunks(file_path, lines_per_chunk)
        for partial_counts in pool.imap_unordered(mapper, chunks, chunksize=1):
            for word, count in partial_counts.items():
                reducer_id = zlib.crc32(word.encode("utf-8")) % reducers
                partitions[reducer_id][word] += count
    final_counts = Counter()
    for partition in partitions:
        final_counts.update(partition)
    return final_counts

def save_counts(counts, output_file):
    sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    with open(output_file, "w", encoding="utf-8") as file:
        for word, count in sorted_counts:
            file.write(f"{word}\t{count}\n")

def main():
    parser = argparse.ArgumentParser(description="MapReduce Word Frequency Counter")
    parser.add_argument("input", help="Input text file")
    parser.add_argument("--output", default="word_counts.txt", help="Output file")
    parser.add_argument("--workers", type=int, default=2, help="Number of mapper processes")
    parser.add_argument("--reducers", type=int, default=4, help="Number of reducer partitions")
    parser.add_argument("--chunk-lines", type=int, default=10000, help="Lines processed by one mapper task")
    parser.add_argument("--top", type=int, default=20, help="Number of top words to display")

    args = parser.parse_args()
    start_time = time.perf_counter()

    counts = map_reduce_word_count(
        file_path=args.input, workers=args.workers, reducers=args.reducers, lines_per_chunk=args.chunk_lines
    )

    elapsed_time = time.perf_counter() - start_time
    save_counts(counts, args.output)

    print("MapReduce completed\n-------------------")
    print(f"Total words: {sum(counts.values()):,}\nUnique words: {len(counts):,}")
    print(f"Execution time: {elapsed_time:.3f} seconds\nOutput file: {args.output}")
    print(f"\nTop {args.top} words:\n-------------------")
    for word, count in counts.most_common(args.top):
        print(f"{word:<20} {count}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
