# AI-Powered Semantic Duplicate Detector

An NLP-based application that determines whether two questions have the same meaning, even when they are written using different words.

The system converts questions into semantic vector representations using Sentence Transformers and compares them using cosine similarity.

## Features

- Compare two questions for semantic similarity
- Sentence embedding-based semantic understanding
- Cosine similarity scoring
- Automatically identifies duplicate and non-duplicate questions
- Validated similarity decision threshold
- Human-readable explanation of the prediction
- Interactive Gradio web interface
- Session similarity visualization
- Example question pairs for quick testing

## How It Works

```text
Input Question A
        +
Input Question B
        ↓
Sentence Transformer
        ↓
Semantic Embeddings
        ↓
Cosine Similarity
        ↓
Validated Threshold
        ↓
Duplicate / Non-Duplicate
