# Description

This repository contains the projects and assignments for the **Modern Information Retrieval course at Sharif University of Technology, taught by Dr. Soleymani.**

The repository is divided into three main phases, transitioning from classical search engines to modern multimodal AI-based retrieval systems.

## Table of Contents

- [Phase 1: Search Engine & Classic IR Pipeline](#-phase-1-search-engine--classic-ir-pipeline) - Building a Goodreads book search engine from scratch.
- [Phase 2: NLP & Representation Learning](#-phase-2-nlp--representation-learning) - Applying ML/NLP techniques for text classification, clustering, and embeddings.
- [Phase 3: Hybrid Multimodal Product Search](#-phase-3-hybrid-multimodal-product-search) - Creating an advanced product search pipeline handling text and image queries.

---

## Phase 1: Search Engine & Classic IR Pipeline

This phase implements a complete Information Retrieval (IR) pipeline for a Goodreads dataset. It features a backend logic engine and a Streamlit-based UI for interactive searching.

- **Concepts Used:**
  - Text Preprocessing (Tokenization, Link removal, Stopwords, Stemming/Lemmatization)
  - Near-Duplicate Detection (MinHashLSH, Shingling)
  - Indexing (Inverted Indexes, Tiered Indexing)
  - Spell Correction (Jaccard Similarity + Normalized TF score)
  - Scoring & Ranking Models (Vector Space Model (VSM), Okapi BM25, Unigram Language Model)
  - Snippet Generation
  - System Evaluation (MAP, NDCG, MRR, Precision, Recall)

- **Run Commands:**
  ```bash
  cd Phase1
  pip install -r requirements.txt
  streamlit run UI/main.py
  ```

---

## Phase 2: NLP & Representation Learning

This phase focuses on natural language processing tasks applied to book datasets, implemented primarily via Jupyter Notebooks.

- **Concepts Used:**
  - Word Embeddings (Word2Vec)
  - Document Classification
  - Text Clustering
  - Advanced Language Models (BERT)

**Note:** Execution is fully notebook-based and you can see its results or run from scratch.

---

## Phase 3: Hybrid Multimodal Product Search

This phase is an advanced project building a realistic product search pipeline for the Amazon Berkeley Objects (ABO) dataset. It supports single and combined text+image queries, fusing different retrieval methods, and generating structured LLM responses.

- **Concepts Used:**
  - Single-Vector Multimodal Dense Embeddings (Image + Text combined inputs)
  - Sparse Neural Embeddings (SPLADE++)
  - Hybrid Candidate Retrieval (Fusion of Dense + Sparse results)
  - Cross-Encoder Fusion and Reranking
  - Structured LLM Answer Generation

- **Run Commands:**
  ```bash
  cd Phase3
  
  # Base dependency synchronization using uv
  uv sync
  
  # (Optional) For embedding, retrieval, and reranking runtimes:
  uv sync --extra stage1-gpu
  
  # Example: Run catalog tests
  uv run python src/tests.py --stage 0 --catalog data/catalog_subset.parquet
  ```
