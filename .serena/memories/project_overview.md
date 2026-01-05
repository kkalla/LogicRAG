# LogicRAG Project Overview

## Purpose
LogicRAG enables structured retrieval without building knowledge graphs on corpora. By constructing query logic dependency graphs to guide structured retrieval adaptively, it enables test-time scaling of graphRAG on large/dynamic knowledge bases. This work has been accepted to **AAAI 2026**.

## Key Features
1. **Logic Dependency Analysis**: Convert complex questions into logical dependency graphs for planning multi-step retrieval
2. **Graph Reasoning Linearization**: Linearize complex graph reasoning into sequential subproblem solution while maintaining logic-coherence
3. **Efficiency**: Efficient scheduling via graph pruning, and context-length optimization via rolling memory
4. **Interpretable Results**: Provides clear reasoning paths and dependency analysis for better explainability

## Tech Stack
- **Language**: Python (>=3.7)
- **ML Framework**: PyTorch (>=1.9.0)
- **Embeddings**: sentence-transformers (>=2.2.0), model: all-MiniLM-L6-v2
- **LLM**: OpenAI API (default: gpt-4o-mini)
- **Utilities**: tqdm, numpy, backoff, ratelimit, python-dotenv, colorama

## Codebase Structure
```
LogicRAG/
├── config/
│   └── config.py           # API keys, model config, rate limiting
├── dataset/                # Dataset files location
├── figs/                   # Figures/diagrams
├── src/
│   ├── models/
│   │   ├── base_rag.py     # Base RAG class with embedding/retrieval
│   │   └── logic_rag.py    # Main LogicRAG implementation
│   ├── utils/
│   │   └── utils.py        # API calls, JSON parsing, evaluation utilities
│   ├── evaluation/
│   │   └── evaluation.py   # Evaluation metrics
│   └── main.py             # CLI entry point
├── cache/                  # Cached embeddings (created at runtime)
├── result/                 # Evaluation results (created at runtime)
└── run.py                  # Main script
```

## Design Patterns
- **Inheritance**: `LogicRAG` extends `BaseRAG`
- **Caching**: Embeddings cached as `.pt` files in `cache/` directory
- **Rate Limiting**: Uses `@limits` decorator from `ratelimit` for API calls
- **Retry Logic**: Uses `@backoff.on_exception` for resilient API calls
- **Graph Algorithm**: Topological sort using DFS for dependency resolution

## Algorithm Flow
1. **Stage 1 - Warm-up**: Simple retrieval + initial analysis to check if question needs complex reasoning
2. **Stage 2 - Dependency Analysis**: If needed, decompose question into logical dependencies
3. **Stage 3 - Topological Sort**: Sort dependencies in resolution order
4. **Stage 4 - Iterative Retrieval**: Resolve each dependency sequentially with rolling memory summary
