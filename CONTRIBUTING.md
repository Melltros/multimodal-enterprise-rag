# Contributing to Multimodal Enterprise RAG

Thank you for your interest in contributing to the **Multimodal Enterprise RAG** platform!

## Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/multimodal-enterprise-rag.git
   cd multimodal-enterprise-rag
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run the Test Suite**
   ```bash
   python -m unittest discover tests
   ```

## Pull Request Guidelines

- Ensure all new features have accompanying unit tests in the `tests/` directory.
- Preserve explicit typing and comprehensive docstrings for public classes and functions.
- Run smoke tests before opening a PR:
  ```bash
  python main.py --mock
  ```
