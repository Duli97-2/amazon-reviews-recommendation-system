# amazon-reviews-recommendation-system

A multi-approach recommendation system built on the Amazon Software Reviews dataset, implementing and comparing three fundamentally different techniques to generate personalised product recommendations at scale.

---

## Overview

This system follows a two-stage **retrieve-then-rank** pipeline. Recommendations are pre-computed offline and stored in Elasticsearch, enabling fast, low-latency retrieval at query time. Three approaches are implemented and evaluated side by side.

| Approach | Method | Infrastructure |
|---|---|---|
| Collaborative Filtering | Alternating Least Squares (ALS) | Apache Spark MLlib |
| Graph-Based CF | Neighbourhood Traversal | Neo4j |
| Content-Based Filtering | TF-IDF + Cosine Similarity | Scikit-learn |

All three methods store their top recommendations per user in **Elasticsearch**, which acts as the unified serving layer across the entire system.

---

## Dataset

- **Source:** Amazon Reviews 2023 – Software Category
- **Raw reviews:** 4,829,124
- **After filtering** (≥3 reviews/user, ≥5 reviews/product): 2,180,215 reviews
- **Unique users:** 409,542 | **Unique products:** 34,871
- **Training set** (pre-Jan 2019): 1,744,172 reviews across 365,365 users
- **Test set:** 202,374 reviews across 84,386 users

---

## Approaches

### 1. ALS Collaborative Filtering

Matrix factorisation on the user-product interaction matrix using Apache Spark MLlib.

- **Parameters:** rank=10, regParam=0.1, maxIter=10
- **Train/Test Split:** Timestamp-based (pre/post January 2019)
- **RMSE:** 1.9183
- **Scale:** 365,365 users processed in 74 distributed batches
- **Storage:** Top-200 recommendations per user stored in Elasticsearch

### 2. Graph-Based Collaborative Filtering

User behaviour modelled as a graph in Neo4j, where edges represent product reviews. Batched Cypher queries traverse user neighbourhoods to find co-purchase patterns.

- **Graph:** 365,365 User nodes + 1,744,172 REVIEWED edges
- **Graph density:** 0.014%
- **Batch size:** 50 users per batch, neighbour LIMIT=100
- **Coverage:** 0.27% of catalogue

### 3. Content-Based Filtering

Builds a TF-IDF profile for each user from their highly-rated products (rating ≥ 4.0) and recommends similar items using cosine similarity.

- **Vectoriser:** TF-IDF, max_features=15,000, ngram_range=(1,2), sublinear_tf=True
- **User profile:** Mean TF-IDF vector of highly-rated products
- **Generation:** Batched sparse matrix multiplication
- **Coverage:** 3.37% of catalogue — highest of all approaches

---

## Evaluation Results

Evaluated on 1,000 sampled test users with relevance threshold of rating ≥ 4.0.

| Approach | RMSE | Precision@10 | Recall@10 | Coverage |
|---|---|---|---|---|
| Popularity Baseline | N/A | 0.0299 | 0.0691 | 0.04% (~14 products) |
| ALS CF | 1.9183 | 0.0001 | 0.0003 | 2.66% (~927 products) |
| Graph CF | N/A | 0.0153 | 0.0265 | 0.27% (~94 products) |
| Content-Based | N/A | 0.0285 | 0.0686 | 3.37% (~1,175 products) |

### Key Findings

- **Content-based filtering** achieves 84× better catalogue coverage than the popularity baseline, surfacing niche software titles that collaborative methods miss
- **Graph CF** is limited by extreme sparsity — with only 0.014% graph density, most users share too few common neighbours for meaningful traversal
- **ALS** near-zero precision/recall is due to a product index mapping issue in Elasticsearch retrieval — the model RMSE of 1.9183 confirms the model itself trained correctly
- **Content-based filtering** is the most effective approach for sparse, single-purchase datasets like Software

---

## Infrastructure

| Component | Purpose |
|---|---|
| Apache Spark (PySpark) | Distributed ALS training |
| Neo4j | Graph database for user-product relationships |
| Elasticsearch 9.x | Pre-computed recommendation store and serving layer |
| Scikit-learn | TF-IDF vectorisation and cosine similarity |
| Python 3.x | Orchestration and evaluation |

---

## Prerequisites

```bash
# Python environment
conda activate scc454_project

# Services required (must be running before executing notebook)
# 1. Elasticsearch on localhost:9200
# 2. Neo4j on bolt://127.0.0.1:7687

# Java (required for Spark)
# JDK 17 — set JAVA_HOME before running
```

---

## File Structure

```
task4/
│
├── task4_recommendation_sys.ipynb          # Software category notebook
├── task4_recommendation_sys_b&p.ipynb      # Beauty & Personal Care notebook
├── task4_recommendation_sys_fashion.ipynb  # Fashion category notebook
├── task1_preprocessing.py                  # Preprocessing pipeline
├── task4_preprocess_bp.py                  # B&P preprocessing
└── task4_preprocess_fashion.py             # Fashion preprocessing
```

---

## Configuration

Key parameters in the notebook — update these to match your local setup:

```python
# File paths
REVIEWS_PATH = 'F:/scc454/t4/Processed/software_final.csv'
PRODUCTS_PATH = 'F:/scc454/t4/Processed/software_products_final.csv'

# Elasticsearch
ES_HOST = 'http://localhost:9200'
RECS_INDEX = 'software_recommendations'

# Neo4j
NEO4J_URI = 'bolt://127.0.0.1:7687'
NEO4J_PASSWORD = 'password'
NEO4J_DATABASE = 'neo4j'

# Spark
JAVA_HOME = r'C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot'
SPARK_TMP = 'F:/scc454/t4/spark_tmp'
```

---

## Limitations

- The Software category is dominated by single purchases, making collaborative signals weak
- Graph CF effectiveness is severely limited by 0.014% graph density
- ALS evaluation metrics are affected by a product index mapping issue in Elasticsearch
- Evaluation performed on Software category only — results may differ for denser categories

---

## Author

Dulitha Vindula Abeysuriya Patabendige  
MSc Data Science — Lancaster University  
SCC.454 Group H — Task 4

