# NIQ TASK1 & TASK2 - Multimodal Data Analytics Pipeline

This repository is part of the NIQ (NielsenIQ) Multimodal Food Product Analytics Challenge. The pipeline integrates both textual and image-based data from the OpenFoodFacts dataset to extract nutritional entities, predict NutriScore, and build a robust, scalable multi-modal data processing system.

---

## 🧩 Tasks Overview

### ✅ TASK 1: **Preprocessing & Extraction**
- Parse raw TSV data from OpenFoodFacts
- Clean and preprocess product text and nutrition fields
- Download product images via AWS or public URLs
- Normalize, resize and prepare image data for modeling
- Extract nutrition and ingredient entities (manually or via LLMs)

### ✅ TASK 2: **Multimodal Analysis & Prediction**
- Fuse text and image features
- Use processed data for NutriScore prediction
- Store results in structured formats (JSONL, PNG, etc.)
- Serve via a local API for per-product analysis