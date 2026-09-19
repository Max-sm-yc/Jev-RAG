# RAG with Jev as Re-Ranker

This project contains a basic RAG pipeline with vector search, re-ranker via TypeSafe AI's Jev, and final output by Muse Spark 1.3

The use of Jev Re-Ranker cut total costs by over 70% and total response time by 72%.

## Analysis

The same search was run with RAG + Jev Re-Ranker + Muse Spark, RAG + Muse Spark Re-Ranker + Muse Spark, and Muse Spark with whole context. Costs include generating embeddings.


RAG + Jev Re-Ranker + Muse Spark

    Total Cost: $0.00122838
    Total Request Time: 62.3s

RAG + Muse Spark Re-Ranker + Muse Spark

    Total Cost: $0.00421838
    Total Request Time: 228.14s

Muse Spark (full context, no RAG)

    Total Cost: $0.0032
    Total Request Time: 10.60s

The search was conducted over ~30,000 tokens with end output tokens being in the 1000-1400 token range.

## Setup instructions
py -3.12 -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

