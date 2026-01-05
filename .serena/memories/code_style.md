# LogicRAG Code Style & Conventions

## Python Style
- **PEP 8**: Generally follows PEP 8 guidelines
- **Type Hints**: Uses `typing` module (List, Dict, Tuple, Any)
- **Docstrings**: Google-style docstrings for classes and methods

## Naming Conventions
- **Classes**: `PascalCase` (e.g., `LogicRAG`, `BaseRAG`)
- **Functions/Methods**: `snake_case` (e.g., `answer_question`, `refine_summary_with_context`)
- **Private Methods**: `_snake_case` prefix (e.g., `_retrieve_with_filter`, `_topological_sort`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `OPENAI_API_KEY`, `DEFAULT_MODEL`)
- **Variables**: `snake_case` (e.g., `info_summary`, `current_query`)

## Code Organization
- **Imports**: Standard library → Third-party → Local imports
- **Class Structure**:
  1. `__init__` method first
  2. Public methods
  3. Private methods (prefixed with `_`)
  4. Static methods (decorated with `@staticmethod`)

## Error Handling
- Uses try-except blocks with logging
- Fallback values on failure (e.g., returns default dict if JSON parse fails)
- Color-coded logging using `colorama` (Fore.RED for errors)

## Logging
- Uses Python `logging` module
- Color-coded output with `colorama`
- `logger.info()` for progress tracking
- `logger.error()` for errors

## API Interaction
- OpenAI API via `openai` package (v1.0+)
- Rate limiting: `@sleep_and_retry @limits(calls=CALLS_PER_MINUTE, period=PERIOD)`
- Retry with exponential backoff: `@backoff.on_exception`

## JSON Parsing
- Custom `fix_json_response()` function handles LLM output quirks
- Handles missing closing braces and truncated fields
- Removes markdown code blocks (```json, ```)

## Configuration
- Uses `python-dotenv` for environment variables
- `.env` file for API keys (not committed to git)
- Central config in `config/config.py`
