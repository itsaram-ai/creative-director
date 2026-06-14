"""TF-IDF retrieval over the Markdown knowledge base.

The corpus is intentionally small, so this module implements sparse TF-IDF
vectors and cosine similarity directly instead of introducing an embedding
model or external vector database.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import string
import math


@dataclass
class Chunk:
    # A chunk is one searchable piece of a knowledge-base file.
    id: str
    text: str
    source_file: str


def load_chunks(knowledge_dir: str | Path) -> list[Chunk]:
    """Read Markdown files and divide their paragraphs into searchable chunks."""
    # Convert a string path into a Path object so we can use methods like glob.
    path = Path(knowledge_dir)
    # Find every Markdown file and sort them to keep chunk order predictable.
    knowledge_files = list(path.glob("*.md"))
    knowledge_files.sort()
    # This list will contain all chunks created from all knowledge files.
    chunks = []
    for file in knowledge_files:
        text = file.read_text(encoding="utf-8")
        # A blank line separates paragraphs in these Markdown files.
        paragraphs = text.split("\n\n")
        # The buffer collects short paragraphs until they reach about 40 words.
        buffer = []
        # Each file has its own chunk numbering, starting from zero.
        chunk_index = 0
        for paragraph in paragraphs:
            # strip() removes spaces and newlines from both ends.
            cleaned_paragraph = paragraph.strip()
            if cleaned_paragraph:
                buffer.append(cleaned_paragraph)
                # Keep paragraph breaks when combining the buffered text.
                combined_buffered_paragraph = "\n\n".join(buffer)
                combined_word_count = len(combined_buffered_paragraph.split())

                # Once the buffer is large enough, turn it into a searchable chunk.
                if combined_word_count >= 40:
                    chunks.append(
                        Chunk(
                            id=f"{file.name}:{chunk_index}",
                            text=combined_buffered_paragraph,
                            source_file=file.name
                        )
                    )
                    chunk_index += 1

                    # Empty the buffer so the next chunk starts fresh.
                    buffer.clear()
        # Keep any text left at the end, even if it did not reach 40 words.
        if buffer:
            combined_buffered_paragraph = "\n\n".join(buffer)
            chunks.append(
                Chunk(
                    id=f"{file.name}:{chunk_index}",
                    text=combined_buffered_paragraph,
                    source_file=file.name
                )
            )

    return chunks
    

STOPWORDS = {"the", "and", "for", "are", "but", "not", "with", "this"}


def tokenize(text: str) -> list[str]:
    """Clean text and return the useful words used for retrieval."""
    # Lowercasing makes words such as "TikTok" and "tiktok" match.
    text = text.lower()
    # Build instructions that delete every character in string.punctuation.
    translation_table = str.maketrans("", "", string.punctuation)
    # Apply those deletion instructions to the text.
    text = text.translate(translation_table)
    # Split on whitespace to produce individual word tokens.
    tokens = text.split()
    filtered_tokens = []
    for t in tokens:
        # Common words and very short words add little value to retrieval.
        if t not in STOPWORDS and len(t) >= 3:
            filtered_tokens.append(t)
    return filtered_tokens


@dataclass
class Index:
    # The original chunks are kept so search can return their text and source.
    chunks: list[Chunk]
    # Each sparse vector maps only the terms in that chunk to TF-IDF weights.
    vectors: list[dict[str, float]]
    # The shared IDF table is reused when vectorising search queries.
    idf: dict[str, float]


def build_index(chunks: list[Chunk]):
    """Build TF-IDF vectors and vocabulary weights for a list of chunks."""
    # Tokenise every chunk once before calculating corpus-wide statistics.
    tokens_per_chunk = []

    for chunk in chunks:
        tokenised_chunk = tokenize(chunk.text)
        tokens_per_chunk.append(tokenised_chunk)

    # Document frequency records how many different chunks contain each term.
    document_frequency = {}
    for chunk_tokens in tokens_per_chunk:
        # A set prevents repeated terms in one chunk from being counted twice.
        unique_terms = set(chunk_tokens)
        for term in unique_terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    
    idf = {}
    # idf formula: idf(term) = log(N / (1 + df(term))), where N is the number of chunks
    # Rare terms receive larger weights because they identify chunks more clearly.
    for term, frequency in document_frequency.items():
        idf[term] = math.log(len(chunks) / (1 + frequency))

    # Build one sparse TF-IDF vector for each chunk.
    vectors = []
    for chunk_tokens in tokens_per_chunk:
        # Avoid dividing by zero if a chunk has no useful tokens.
        if not chunk_tokens:
            vectors.append({})
            continue

        # Count how many times each term appears in this specific chunk.
        term_counts = {}
        for term in chunk_tokens:
            term_counts[term] = term_counts.get(term, 0) + 1

        vector = {}
        # TF-IDF weight calculation
        for term, count in term_counts.items():
            # TF measures how common the term is within this chunk.
            term_frequency = count / len(chunk_tokens)
            # Multiplying TF by IDF balances local frequency with overall rarity.
            weight = term_frequency * idf[term]
            vector[term] = weight

        vectors.append(vector)

    return Index(chunks=chunks, vectors=vectors, idf=idf)


def top_k(index, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
    """Return the k chunks whose TF-IDF vectors best match the query."""
    # Process the query with the same tokenizer used for the knowledge chunks.
    query_tokens = tokenize(query)
    query_counts = {}

    # An empty query cannot match any chunk.
    if not query_tokens:
        return []

    # Count query terms, ignoring words that do not exist in the index.
    for term in query_tokens:
        if term in index.idf:
            query_counts[term] = query_counts.get(term, 0) + 1

    # Convert the query into a sparse TF-IDF vector using the index's IDF table.
    query_vector = {}
    for term, count in query_counts.items():
        term_frequency = count / len(query_tokens)
        weight = term_frequency * index.idf[term]
        query_vector[term] = weight

    # Known vocabulary is required to compare the query with the chunks.
    if not query_vector:
        return []
    
    # The norm is the geometric length of the query vector.
    query_norm = math.sqrt(
        sum(weight **2 for weight in query_vector.values())
        )
    
    results = []
    # zip pairs each original chunk with its corresponding TF-IDF vector.
    for chunk, chunk_vector in zip(index.chunks, index.vectors):
        # The dot product measures overlap between weighted query and chunk terms.
        dot_product = sum(
            query_weight * chunk_vector.get(term, 0)
            for term, query_weight in query_vector.items()
        )

        # Calculate the geometric length of the current chunk vector.
        chunk_norm = math.sqrt(
            sum(weight ** 2 for weight in chunk_vector.values())
        )

        # Cosine similarity measures vector direction, not raw text length.
        # A higher score means the query and chunk use similar important terms.
        if chunk_norm == 0 or query_norm == 0:
            score = 0.0
        else:
            score = dot_product / (query_norm * chunk_norm)
        results.append((chunk, score))


    # Sort by the score at tuple index 1, placing the best matches first.
    results.sort(key=lambda result: result[1], reverse=True)

    # Return no more than the requested number of highest-scoring matches.
    return results[:k]
