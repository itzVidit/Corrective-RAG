# 🔍 Corrective RAG (CRAG) — Paper Implementation

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-black.svg)](https://platform.openai.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange.svg)](https://faiss.ai/)
[![Paper](https://img.shields.io/badge/arXiv-2401.15884-red.svg)](https://arxiv.org/abs/2401.15884)

> Implementation of **Corrective Retrieval Augmented Generation (CRAG)** — a technique that improves standard RAG by evaluating retrieved documents and deciding whether to use them, fix them with web search, or discard them entirely before generating an answer.

---

## 📄 Paper

This implementation is based on:

**Corrective Retrieval Augmented Generation**  
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling  
arXiv:2401.15884 · [https://arxiv.org/abs/2401.15884](https://arxiv.org/abs/2401.15884)

> *"A lightweight retrieval evaluator is designed to assess the overall quality of retrieved documents for a query, returning a confidence degree based on which different knowledge retrieval actions can be triggered."*

---

## 💡 The Core Idea

Standard RAG blindly trusts whatever documents it retrieves. CRAG fixes this with a simple but powerful idea: **evaluate the retrieved docs before using them**.

Each retrieved document gets a relevance score. Based on the scores, the pipeline takes one of three paths:

| Verdict | Condition | Action |
|---------|-----------|--------|
| **CORRECT** | At least one doc scores `> 0.7` | Use internal docs directly |
| **INCORRECT** | All docs score `< 0.3` | Discard internal docs, search the web |
| **AMBIGUOUS** | Neither of the above | Combine internal docs + web results |

In all three cases, a **knowledge refinement step** strips the context down to only the sentences that are directly relevant to the question before generating the final answer.

---

## 🏗️ Pipeline Architecture

The pipeline is built as a **LangGraph state graph (DAG)**:

```
          ┌──────────────────────────────┐
          │         User Question        │
          └──────────────┬───────────────┘
                         │
                  ┌──────▼──────┐
                  │   retrieve  │  FAISS similarity search (top-4 chunks)
                  └──────┬──────┘
                         │
                 ┌────────▼────────┐
                 │  eval_each_doc  │  LLM scores each chunk [0.0 → 1.0]
                 └────────┬────────┘
                          │
          ┌───────────────┼───────────────┐
          │                               │
    [CORRECT]                   [INCORRECT / AMBIGUOUS]
          │                               │
          │                    ┌──────────▼──────────┐
          │                    │   rewrite_query     │  LLM rewrites question
          │                    └──────────┬──────────┘  into a web search query
          │                               │
          │                    ┌──────────▼──────────┐
          │                    │     web_search      │  Tavily search (top-5)
          │                    └──────────┬──────────┘
          │                               │
          └───────────────┬───────────────┘
                          │
                   ┌──────▼──────┐
                   │   refine    │  Decompose context into sentences →
                   └──────┬──────┘  LLM filters to only relevant ones
                          │
                   ┌──────▼──────┐
                   │  generate   │  Answer using refined context only
                   └──────┬──────┘
                          │
                      [Answer]
```

---

## 🔧 How Each Part Works

### 1. Retrieval
PDF documents are loaded, chunked (`chunk_size=900`, `overlap=150`), embedded with `text-embedding-3-large`, and stored in a FAISS vector store. Top-4 similar chunks are retrieved per query.

### 2. Document Evaluation
Each retrieved chunk is scored `[0.0, 1.0]` by the LLM with a strict system prompt. The scoring determines the verdict:
- `score > 0.7` on any chunk → **CORRECT**
- All scores `< 0.3` → **INCORRECT**
- Anything else → **AMBIGUOUS**

### 3. Query Rewriting (non-CORRECT paths)
The original question is rewritten into a short, keyword-focused web search query. If the question implies recency, the rewriter adds a time constraint like `(last 30 days)`.

### 4. Web Search
Tavily search fetches up to 5 web results, each stored as a `Document` with title, URL, and content.

### 5. Knowledge Refinement
The selected context (internal, web, or both) is decomposed into individual sentences. An LLM judge then filters each sentence — keeping only those that directly help answer the question. This is the **decompose-then-recompose** step from the paper.

### 6. Answer Generation
The final answer is generated using only the refined context. If context is empty, the model says `"I don't know."` — no hallucination.

---

## 📁 File Structure

```
C_RAG/
├── 6_ambiguous.ipynb   ← Full CRAG implementation (notebook)
├── 2401.15884v3.pdf    ← Original research paper
└── documents/          ← Put your PDFs here (book1.pdf, book2.pdf, book3.pdf)
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- Tavily API key (free tier available at [tavily.com](https://tavily.com))

### Install dependencies

```bash
pip install langchain langchain-community langchain-openai langgraph faiss-cpu pypdf python-dotenv tavily-python
```

### Configure keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### Add your documents

Place your PDF files in a `documents/` folder:

```
documents/
├── book1.pdf
├── book2.pdf
└── book3.pdf
```

### Run

Open `6_ambiguous.ipynb` in Jupyter and run all cells.

---

## 🧪 Example

```python
res = app.invoke({
    "question": "Batch normalization vs layer normalization",
    ...
})

print(res["verdict"])    # AMBIGUOUS
print(res["web_query"])  # "Batch normalization vs layer normalization"
print(res["answer"])     # Final answer using refined context
```

**Sample output:**
```
VERDICT: AMBIGUOUS
REASON: No chunk scored > 0.7, but not all were < 0.3.
WEB_QUERY: Batch normalization vs layer normalization

OUTPUT:
 Batch normalization normalizes over the mini-batch, making it effective
 for CNNs. Layer normalization normalizes per example, making it better
 suited for RNNs and variable-length inputs...
```

---

## 🧰 Tech Stack

| Technology | Role |
|------------|------|
| **LangGraph** | DAG orchestration, conditional routing, state management |
| **LangChain** | Document loaders, prompt templates, retriever abstraction |
| **OpenAI** | LLM (`gpt-4o-mini`) + embeddings (`text-embedding-3-large`) |
| **FAISS** | Local vector store for document chunks |
| **Tavily** | Web search API for the INCORRECT/AMBIGUOUS paths |
| **Pydantic** | Structured LLM output (`DocEvalScore`, `KeepOrDrop`, `WebQuery`) |

---

## 📌 Key Differences from Basic RAG

| Feature | Basic RAG | CRAG (this impl.) |
|---------|-----------|-------------------|
| Document quality check | ❌ | ✅ Score-based evaluator |
| Web search fallback | ❌ | ✅ Triggered automatically |
| Ambiguous case handling | ❌ | ✅ Merges internal + web |
| Context refinement | ❌ | ✅ Sentence-level filtering |
| Query rewriting for search | ❌ | ✅ LLM rewrites the query |

---

## 📚 Reference

```bibtex
@article{yan2024corrective,
  title   = {Corrective Retrieval Augmented Generation},
  author  = {Shi-Qi Yan and Jia-Chen Gu and Yun Zhu and Zhen-Hua Ling},
  journal = {arXiv preprint arXiv:2401.15884},
  year    = {2024},
  url     = {https://arxiv.org/abs/2401.15884}
}
```

---
