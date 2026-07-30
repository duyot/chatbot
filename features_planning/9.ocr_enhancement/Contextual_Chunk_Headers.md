# `Semantic Chunking` for Document Processing

## Overview

This code implements a semantic chunking approach for processing and retrieving information from PDF documents, [first proposed by Greg Kamradt](https://youtu.be/8OJC21T2SL4?t=1933) and subsequently [implemented in LangChain](https://python.langchain.com/docs/how_to/semantic-chunker/). Unlike traditional methods that split text based on fixed character or word counts, semantic chunking aims to create more meaningful and context-aware text segments.

## Motivation

Traditional text splitting methods often break documents at arbitrary points, potentially disrupting the flow of information and context. Semantic chunking addresses this issue by attempting to split text at more natural breakpoints, preserving semantic coherence within each chunk.

## Key Components

1. PDF processing and text extraction
2. Semantic chunking using LangChain's SemanticChunker
3. Vector store creation using FAISS and OpenAI embeddings
4. Retriever setup for querying the processed documents

## Method Details

### Document Preprocessing

1. The PDF is read and converted to a string using a custom `read_pdf_to_string` function.

### Semantic Chunking

1. Utilizes LangChain's `SemanticChunker` with OpenAI embeddings.
2. Three breakpoint types are available:
   - 'percentile': Splits at differences greater than the X percentile.
   - 'standard_deviation': Splits at differences greater than X standard deviations.
   - 'interquartile': Uses the interquartile distance to determine split points.
3. In this implementation, the 'percentile' method is used with a threshold of 90.

### Vector Store Creation

1. OpenAI embeddings are used to create vector representations of the semantic chunks.
2. A FAISS vector store is created from these embeddings for efficient similarity search.

### Retriever Setup

1. A retriever is configured to fetch the top 2 most relevant chunks for a given query.

## Key Features

1. Context-Aware Splitting: Attempts to maintain semantic coherence within chunks.
2. Flexible Configuration: Allows for different breakpoint types and thresholds.
3. Integration with Advanced NLP Tools: Uses OpenAI embeddings for both chunking and retrieval.

## Benefits of this Approach

1. Improved Coherence: Chunks are more likely to contain complete thoughts or ideas.
2. Better Retrieval Relevance: By preserving context, retrieval accuracy may be enhanced.
3. Adaptability: The chunking method can be adjusted based on the nature of the documents and retrieval needs.
4. Potential for Better Understanding: LLMs or downstream tasks may perform better with more coherent text segments.

## Implementation Details

1. Uses OpenAI's embeddings for both the semantic chunking process and the final vector representations.
2. Employs FAISS for creating an efficient searchable index of the chunks.
3. The retriever is set up to return the top 2 most relevant chunks, which can be adjusted as needed.

## Example Usage

The code includes a test query: "What is the main cause of climate change?". This demonstrates how the semantic chunking and retrieval system can be used to find relevant information from the processed document.

## Conclusion

Semantic chunking represents an advanced approach to document processing for retrieval systems. By attempting to maintain semantic coherence within text segments, it has the potential to improve the quality of retrieved information and enhance the performance of downstream NLP tasks. This technique is particularly valuable for processing long, complex documents where maintaining context is crucial, such as scientific papers, legal documents, or comprehensive reports.