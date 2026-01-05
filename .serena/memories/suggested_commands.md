# LogicRAG Suggested Commands

## Installation & Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with:
OPENAI_API_KEY=your_api_key_here
```

## Running the Project

### Evaluate on Dataset
```bash
python run.py --model logic-rag --dataset path/to/dataset.json --corpus path/to/corpus.json --max-rounds 5 --top-k 3
```

### Run Single Question
```bash
python run.py --model logic-rag --question "Your question here" --corpus path/to/corpus.json --max-rounds 5 --top-k 3
```

### Limit Question Count
```bash
python run.py --model logic-rag --dataset path/to/dataset.json --corpus path/to/corpus.json --limit 20
# Set --limit 0 to process all questions
```

## Key Command Options
- `--model`: Model type (e.g., `logic-rag`)
- `--dataset`: Path to dataset JSON file
- `--corpus`: Path to corpus JSON file
- `--question`: Single question to answer
- `--max-rounds`: Maximum reasoning rounds (default: 3)
- `--top-k`: Number of contexts to retrieve (default: 5)
- `--limit`: Number of questions to evaluate (default: 20, 0 = all)

## Configuration
Edit `config/config.py` for:
- `DEFAULT_MODEL`: LLM model (default: gpt-4o-mini)
- `EMBEDDING_MODEL`: Embedding model (default: all-MiniLM-L6-v2)
- `CALLS_PER_MINUTE`: API rate limit (default: 20)
- `CACHE_DIR`: Cache directory (default: cache)

## System Commands (Darwin/macOS)
```bash
# List files
ls -la

# Find files
find . -name "*.py"

# Search in files
grep -r "pattern" .

# Git operations
git status
git add .
git commit -m "message"
git push

# Python
python -m pytest  # (if tests are added)
python -m pip list
```

## Development Notes
- **No test suite currently exists** - no pytest/unittest files found
- **No linting configuration** - no .flake8, .pylintrc, or black.toml
- **No formatting config** - consider adding black, ruff, or similar
- Cache is auto-generated in `cache/` directory
- Results saved to `result/` directory
