# Contributing Guidelines

Guidelines for collaborating on the IIDS repository.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Adityaaop/IIDS.git
   cd IIDS
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   # .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Follow standard Python conventions (PEP 8) and keep functions focused.

3. Run the test suite before submitting:
   ```bash
   pytest tests/ -v
   flake8 backend/ model/
   ```

4. Commit your changes using conventional prefixes:
   - `feat:` New features
   - `fix:` Bug fixes
   - `test:` Unit or integration test additions
   - `docs:` Documentation updates
   - `ci:` Docker or CI/CD adjustments

5. Push your branch and open a Pull Request against `main`.
