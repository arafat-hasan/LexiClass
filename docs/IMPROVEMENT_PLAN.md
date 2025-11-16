# LexiClass Professional Improvement Plan

**Date:** October 30, 2025
**Version:** 1.0
**Status:** Planning Phase

## Executive Summary

LexiClass is a well-architected document classification toolkit (~2000 LOC) with solid protocol-based design and good separation of concerns. However, it lacks critical production-ready infrastructure including testing, CI/CD, and formal documentation. This plan outlines a structured approach to transform LexiClass into a professional, production-ready open-source project.

**Key Strengths:**
- Clean protocol-based architecture with good extensibility
- Memory-efficient streaming implementations
- Solid configuration system via environment variables
- Good CLI interface with Typer
- Comprehensive logging infrastructure

**Critical Gaps:**
- **No tests** (0% coverage)
- No CI/CD pipeline
- No type checking enforcement
- Incomplete documentation
- No custom exception hierarchy
- Not prepared for PyPI publication

---

## Current State Analysis

### Codebase Metrics
- **Lines of Code:** ~2,000 (excluding dependencies)
- **Modules:** 13 core modules + CLI + datasets
- **Test Coverage:** 0%
- **Type Hints:** Present but not validated
- **Documentation:** README and CLAUDE.md exist, no API docs

### Architecture Assessment

**Strengths:**
1. **Protocol-Based Design:** Clean interfaces in `interfaces.py` allow easy extension
2. **Plugin System:** Simple registry pattern for tokenizers and features
3. **Streaming Support:** Two-pass index building avoids memory issues
4. **Configuration:** Environment-based config with sensible defaults
5. **Memory Management:** Adaptive batching and memory monitoring

**Weaknesses:**
1. **No Abstraction Layers:** Direct coupling to Gensim and sklearn in places
2. **Error Handling:** Generic exceptions make debugging harder
3. **Commented Code:** Dead code in `classifier.py:39-73`
4. **Inconsistent Validation:** Some methods validate input, others don't
5. **Flat Module Structure:** Will become harder to navigate as project grows

### Dependencies Analysis

**Current Stack:**
- Core: numpy, scipy, scikit-learn, gensim
- CLI: typer, beautifulsoup4
- Optional: PyICU, datasets, python-dotenv, psutil

**Issues:**
- No dependency pinning (no lock file)
- No separation of dev/test/prod dependencies
- No automated dependency updates

---

## Improvement Plan by Priority

## 🔴 **CRITICAL PRIORITY**

### 1. Testing Infrastructure (Highest Priority)

**Problem:** Cannot verify correctness, refactoring is risky, hard to onboard contributors

**Solution:**

#### 1.1 Setup Test Framework
```bash
# Add to pyproject.toml
[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-xdist>=3.0",  # parallel testing
    "pytest-timeout>=2.0",
    "pytest-mock>=3.0",
]
```

#### 1.2 Create Test Structure
```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_tokenization.py       # ICUTokenizer tests
│   ├── test_features.py           # FeatureExtractor tests
│   ├── test_encoding.py           # BinaryClassEncoder, MultiClassEncoder
│   ├── test_index.py              # DocumentIndex tests
│   ├── test_classifier.py         # SVMDocumentClassifier tests
│   ├── test_io.py                 # DocumentLoader, load_labels
│   ├── test_evaluation.py         # Evaluation metrics tests
│   ├── test_config.py             # Settings tests
│   └── test_plugins.py            # Plugin registry tests
├── integration/
│   ├── __init__.py
│   ├── test_end_to_end.py         # Full pipeline tests
│   └── test_cli.py                # CLI command tests
├── fixtures/
│   ├── __init__.py
│   ├── sample_data/
│   │   ├── doc1.txt
│   │   ├── doc2.txt
│   │   └── labels.tsv
│   └── sample_outputs/
└── performance/
    └── test_benchmarks.py          # Performance regression tests
```

#### 1.3 Coverage Goals
- **Phase 1:** 50% coverage (core functionality)
- **Phase 2:** 70% coverage (all public APIs)
- **Phase 3:** 85%+ coverage (production-ready)

#### 1.4 Test Configuration
```toml
# Add to pyproject.toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=lexiclass",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "-v",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

**Estimated Effort:** 2-3 weeks
**Success Criteria:** 70%+ code coverage, all public APIs tested

---

### 2. CI/CD Pipeline

**Problem:** No automation, no quality gates, manual testing burden

**Solution:**

#### 2.1 GitHub Actions Workflow

Create `.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [ master, develop ]
  pull_request:
    branches: [ master ]

jobs:
  test:
    name: Test Python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -e ".[test,icu]"

      - name: Run tests
        run: pytest --cov --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        if: matrix.python-version == '3.11'

  lint:
    name: Lint and Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install ruff mypy pylint
          pip install -e .

      - name: Run Ruff
        run: ruff check src/

      - name: Run Ruff format check
        run: ruff format --check src/

      - name: Run mypy
        run: mypy src/lexiclass

      - name: Run pylint
        run: pylint src/lexiclass --exit-zero  # Warning only initially

  build:
    name: Build Package
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install build tools
        run: pip install build twine

      - name: Build package
        run: python -m build

      - name: Check package
        run: twine check dist/*
```

#### 2.2 Pre-commit Hooks

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: check-toml
      - id: debug-statements

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]
```

**Installation:**
```bash
pip install pre-commit
pre-commit install
```

**Estimated Effort:** 1 week
**Success Criteria:** Green CI on all PRs, pre-commit hooks enforced

---

### 3. Type Checking with mypy

**Problem:** Type hints exist but not validated, potential runtime errors

**Solution:**

#### 3.1 mypy Configuration

Add to `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Start lenient, increase later
disallow_any_generics = false
disallow_subclassing_any = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
check_untyped_defs = true

# Per-module options (gradually increase strictness)
[[tool.mypy.overrides]]
module = "lexiclass.interfaces"
disallow_untyped_defs = true
disallow_any_generics = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

#### 3.2 Add py.typed Marker

Create `src/lexiclass/py.typed`:
```
# PEP 561 marker file
```

Update `.gitignore` to not ignore it:
```gitignore
# Allow py.typed
!src/lexiclass/py.typed
```

#### 3.3 Fix Type Issues
- Add missing type annotations
- Fix Protocol implementations
- Add TypedDict for complex dict structures
- Use `typing.Protocol` consistently

**Estimated Effort:** 1 week
**Success Criteria:** mypy passes with no errors, 90%+ type coverage

---

## 🟠 **HIGH PRIORITY**

### 4. Code Quality & Linting

**Problem:** Inconsistent code style, no modern linting

**Solution:**

#### 4.1 Adopt Ruff (Modern All-in-One Linter)

Add to `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
target-version = "py39"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "TID",    # flake8-tidy-imports
    "PTH",    # flake8-use-pathlib
    "RUF",    # Ruff-specific rules
]
ignore = [
    "E501",   # Line too long (handled by formatter)
    "B008",   # Do not perform function call in argument defaults
    "B905",   # zip without strict parameter
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Unused imports in __init__.py
"tests/**/*.py" = ["S101"]  # Allow assert in tests

[tool.ruff.lint.isort]
known-first-party = ["lexiclass"]

[tool.ruff.format]
quote-style = "single"
indent-style = "space"
line-ending = "auto"
```

#### 4.2 Clean Up Code Smells

**Immediate fixes:**
1. Remove dead code in `classifier.py:39-73` (commented build_index)
2. Fix bare except in `classifier.py:218`
3. Refactor hacky type shim in `io.py:64`
4. Break down long methods (>50 lines)
5. Add docstrings to all modules

**Estimated Effort:** 1 week
**Success Criteria:** Ruff passes with no warnings, code is consistently formatted

---

### 5. Custom Exception Hierarchy

**Problem:** Generic `ValueError` everywhere makes error handling difficult

**Solution:**

#### 5.1 Create Exception Module

Create `src/lexiclass/exceptions.py`:
```python
"""Custom exceptions for LexiClass."""

from __future__ import annotations


class LexiClassError(Exception):
    """Base exception for all LexiClass errors."""


class ConfigurationError(LexiClassError):
    """Raised when configuration is invalid."""


class IndexError(LexiClassError):
    """Base class for index-related errors."""


class IndexNotBuiltError(IndexError):
    """Raised when attempting to use an index that hasn't been built."""


class IndexNotFoundError(IndexError):
    """Raised when attempting to load an index that doesn't exist."""


class DocumentNotFoundError(IndexError):
    """Raised when a document ID is not found in the index."""


class ModelError(LexiClassError):
    """Base class for model-related errors."""


class ModelNotTrainedError(ModelError):
    """Raised when attempting to use a model that hasn't been trained."""


class ModelNotFoundError(ModelError):
    """Raised when attempting to load a model that doesn't exist."""


class InvalidModelError(ModelError):
    """Raised when a model file is corrupted or invalid."""


class DataError(LexiClassError):
    """Base class for data-related errors."""


class InvalidDocumentError(DataError):
    """Raised when document format is invalid."""


class InvalidLabelError(DataError):
    """Raised when label format is invalid."""


class InsufficientDataError(DataError):
    """Raised when there's not enough data for training/evaluation."""


class PluginError(LexiClassError):
    """Base class for plugin-related errors."""


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin is not found."""


class PluginRegistrationError(PluginError):
    """Raised when plugin registration fails."""
```

#### 5.2 Replace Generic Exceptions

Update throughout codebase:
- `ValueError("Document index must be built")` → `IndexNotBuiltError(...)`
- `ValueError("Classifier must be trained")` → `ModelNotTrainedError(...)`
- `ValueError(f"Document {doc_id} not in index")` → `DocumentNotFoundError(...)`

**Estimated Effort:** 2-3 days
**Success Criteria:** All custom exceptions used, better error messages

---

### 6. Documentation

**Problem:** No API documentation, limited examples

**Solution:**

#### 6.1 Add Documentation Framework

Choose **MkDocs Material** (modern, beautiful):
```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
```

Create `mkdocs.yml`:
```yaml
site_name: LexiClass Documentation
site_description: Extensible document classification toolkit
site_author: Arafat Hasan
repo_url: https://github.com/arafat/lexiclass
repo_name: arafat/lexiclass

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.tabs
    - search.suggest
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            show_root_heading: true

nav:
  - Home: index.md
  - Getting Started:
      - Installation: getting-started/installation.md
      - Quick Start: getting-started/quickstart.md
      - Configuration: getting-started/configuration.md
  - User Guide:
      - Building Indexes: guide/building-indexes.md
      - Training Models: guide/training.md
      - Making Predictions: guide/predictions.md
      - Similarity Search: guide/similarity.md
      - Evaluation: guide/evaluation.md
  - API Reference:
      - Classifier: api/classifier.md
      - Index: api/index.md
      - Features: api/features.md
      - Tokenization: api/tokenization.md
      - Interfaces: api/interfaces.md
  - Development:
      - Contributing: development/contributing.md
      - Architecture: development/architecture.md
      - Plugin Development: development/plugins.md
  - About:
      - Changelog: about/changelog.md
      - License: about/license.md
```

#### 6.2 Create Documentation Files

```
docs/
├── index.md                        # Landing page
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── configuration.md
├── guide/
│   ├── building-indexes.md
│   ├── training.md
│   ├── predictions.md
│   ├── similarity.md
│   └── evaluation.md
├── api/
│   ├── classifier.md               # Auto-generated from docstrings
│   ├── index.md
│   ├── features.md
│   ├── tokenization.md
│   └── interfaces.md
├── development/
│   ├── contributing.md
│   ├── architecture.md
│   └── plugins.md
└── about/
    ├── changelog.md
    └── license.md
```

#### 6.3 Add Complete Docstrings

Example for `classifier.py`:
```python
class SVMDocumentClassifier:
    """End-to-end SVM-based document classifier with similarity index.

    This classifier orchestrates the complete pipeline from document
    indexing through training and prediction. It supports binary,
    multi-class, and multi-label classification using Linear SVM.

    Attributes:
        tokenizer: Tokenizer for converting text to tokens
        feature_extractor: Extractor for creating document features
        document_index: Index for similarity search
        is_multilabel: Whether this is a multi-label classifier
        classifier: Trained SVM model (sklearn)
        encoder: Label encoder (binary, multi-class, or multi-label)
        is_fitted: Whether the model has been trained
        index_built: Whether the document index has been built

    Examples:
        >>> from lexiclass.classifier import SVMDocumentClassifier
        >>> from lexiclass.io import DocumentLoader, load_labels
        >>>
        >>> # Build index
        >>> clf = SVMDocumentClassifier()
        >>> def stream_factory():
        ...     return DocumentLoader.iter_documents_from_directory("/path/to/texts")
        >>> clf.build_index(index_path="my_index", document_stream_factory=stream_factory)
        >>>
        >>> # Train
        >>> labels = load_labels("labels.tsv")
        >>> clf.train(labels)
        >>> clf.save_model("model.pkl", index_path="my_index")
        >>>
        >>> # Predict
        >>> docs = DocumentLoader.load_documents_from_directory("/path/to/texts")
        >>> preds = clf.predict(docs)

    See Also:
        - :class:`~lexiclass.index.DocumentIndex`: For similarity search
        - :class:`~lexiclass.features.FeatureExtractor`: For feature extraction
        - :class:`~lexiclass.tokenization.ICUTokenizer`: For tokenization
    """
```

#### 6.4 Add Project Metadata Files

**CONTRIBUTING.md:**
```markdown
# Contributing to LexiClass

## Development Setup
...

## Code Style
...

## Testing
...

## Pull Request Process
...
```

**CHANGELOG.md:**
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite
- CI/CD pipeline
- Full API documentation

### Changed
- Improved error handling with custom exceptions

### Fixed
- Type annotations throughout codebase

## [0.1.0] - 2024-10-30

### Added
- Initial release
- SVM-based classification
- Document similarity search
- Streaming index building
```

**CODE_OF_CONDUCT.md:** (Use Contributor Covenant)

#### 6.5 Update README with Badges

```markdown
# LexiClass

[![CI](https://github.com/arafat/lexiclass/workflows/CI/badge.svg)](https://github.com/arafat/lexiclass/actions)
[![codecov](https://codecov.io/gh/arafat/lexiclass/branch/master/graph/badge.svg)](https://codecov.io/gh/arafat/lexiclass)
[![PyPI version](https://badge.fury.io/py/lexiclass.svg)](https://badge.fury.io/py/lexiclass)
[![Python Versions](https://img.shields.io/pypi/pyversions/lexiclass.svg)](https://pypi.org/project/lexiclass/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
```

**Estimated Effort:** 2 weeks
**Success Criteria:** Complete API docs, 5+ examples, comprehensive guides

---

### 7. Dependency Management

**Problem:** No reproducible builds, no dependency updates

**Solution:**

#### 7.1 Option A: Requirements Files

Create `requirements/` directory:
```
requirements/
├── base.txt              # Production dependencies
├── dev.txt               # Development tools (inherits base)
├── test.txt              # Testing dependencies (inherits base)
└── docs.txt              # Documentation dependencies
```

**base.txt:**
```
numpy>=1.22,<2.0
scipy>=1.8,<2.0
scikit-learn>=1.1,<2.0
gensim>=4.3,<5.0
beautifulsoup4>=4.12
typer[all]>=0.9
psutil>=5.9
```

**dev.txt:**
```
-r base.txt
ruff>=0.1.0
mypy>=1.7
pylint>=3.0
pre-commit>=3.5
```

**test.txt:**
```
-r base.txt
pytest>=7.0
pytest-cov>=4.0
pytest-xdist>=3.0
pytest-timeout>=2.0
pytest-mock>=3.0
```

#### 7.2 Option B: Poetry (Recommended)

```bash
poetry init
poetry add numpy scipy scikit-learn gensim beautifulsoup4 typer psutil
poetry add --group dev ruff mypy pylint pre-commit
poetry add --group test pytest pytest-cov pytest-xdist
poetry lock
```

Creates `poetry.lock` for reproducible builds.

#### 7.3 Dependabot Configuration

Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "arafat"
    labels:
      - "dependencies"
    commit-message:
      prefix: "chore"
      include: "scope"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
```

**Estimated Effort:** 3-4 days
**Success Criteria:** Reproducible builds, automated dependency updates

---

## 🟡 **MEDIUM PRIORITY**

### 8. Module Exports & Public API

**Problem:** Unclear what's public API vs internal

**Solution:**

#### 8.1 Add __all__ Definitions

Example for `classifier.py`:
```python
"""Document classification with SVM."""

from __future__ import annotations

__all__ = ["SVMDocumentClassifier"]

# ... rest of code
```

#### 8.2 Update Main __init__.py

`src/lexiclass/__init__.py`:
```python
"""LexiClass: Extensible document classification toolkit."""

from __future__ import annotations

from .classifier import SVMDocumentClassifier
from .index import DocumentIndex
from .features import FeatureExtractor
from .tokenization import ICUTokenizer
from .io import DocumentLoader, load_labels
from .evaluation import (
    EvaluationMetrics,
    evaluate_predictions,
    load_predictions,
    load_ground_truth,
)
from .exceptions import (
    LexiClassError,
    IndexNotBuiltError,
    ModelNotTrainedError,
    ConfigurationError,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "SVMDocumentClassifier",
    "DocumentIndex",
    "FeatureExtractor",
    "ICUTokenizer",
    # IO utilities
    "DocumentLoader",
    "load_labels",
    # Evaluation
    "EvaluationMetrics",
    "evaluate_predictions",
    "load_predictions",
    "load_ground_truth",
    # Exceptions
    "LexiClassError",
    "IndexNotBuiltError",
    "ModelNotTrainedError",
    "ConfigurationError",
    # Version
    "__version__",
]
```

**Estimated Effort:** 1 day
**Success Criteria:** Clear public API, all modules have __all__

---

### 9. Packaging Improvements

**Problem:** Not ready for PyPI publication

**Solution:**

#### 9.1 Enhanced pyproject.toml

```toml
[project]
name = "lexiclass"
version = "0.1.0"
description = "Extensible document classification toolkit (SVM-based) with similarity search"
readme = "README.md"
requires-python = ">=3.9"
license = { text = "MIT" }
authors = [
    { name = "Arafat Hasan", email = "your.email@example.com" }
]
maintainers = [
    { name = "Arafat Hasan", email = "your.email@example.com" }
]
keywords = [
    "nlp",
    "classification",
    "svm",
    "text-classification",
    "similarity-search",
    "gensim",
    "sklearn",
    "machine-learning",
    "document-classification",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Text Processing :: Linguistic",
    "Typing :: Typed",
]

dependencies = [
    "numpy>=1.22,<2.0",
    "scipy>=1.8,<2.0",
    "scikit-learn>=1.1,<2.0",
    "gensim>=4.3,<5.0",
    "beautifulsoup4>=4.12",
    "typer[all]>=0.9",
    "psutil>=5.9",
]

[project.optional-dependencies]
icu = ["PyICU>=2.11"]
dotenv = ["python-dotenv>=1.0"]
datasets = ["datasets>=2.12"]
dev = [
    "ruff>=0.1.0",
    "mypy>=1.7",
    "pylint>=3.0",
    "pre-commit>=3.5",
]
test = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-xdist>=3.0",
    "pytest-timeout>=2.0",
    "pytest-mock>=3.0",
]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.24",
]
all = [
    "lexiclass[icu,dotenv,datasets,dev,test,docs]",
]

[project.urls]
Homepage = "https://github.com/arafat/lexiclass"
Documentation = "https://lexiclass.readthedocs.io"
Repository = "https://github.com/arafat/lexiclass"
"Bug Tracker" = "https://github.com/arafat/lexiclass/issues"
Changelog = "https://github.com/arafat/lexiclass/blob/master/CHANGELOG.md"

[project.scripts]
lexiclass = "lexiclass.cli.main:app"
```

#### 9.2 Version Management

Consider using `setuptools-scm` or `bump2version`:
```toml
[build-system]
requires = ["setuptools>=61.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools_scm]
write_to = "src/lexiclass/_version.py"
```

#### 9.3 Release Checklist

Create `RELEASE.md`:
```markdown
# Release Checklist

- [ ] Update CHANGELOG.md
- [ ] Update version in pyproject.toml
- [ ] Run full test suite: `pytest`
- [ ] Run linters: `ruff check src/`
- [ ] Run type checker: `mypy src/`
- [ ] Build package: `python -m build`
- [ ] Test installation: `pip install dist/lexiclass-*.whl`
- [ ] Create git tag: `git tag -a v0.1.0 -m "Release 0.1.0"`
- [ ] Push tag: `git push origin v0.1.0`
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Create GitHub release with notes
```

**Estimated Effort:** 2-3 days
**Success Criteria:** Successfully published to PyPI, installable via pip

---

### 10. Logging Improvements

**Problem:** Missing structured logging features

**Solution:**

#### 10.1 Add Correlation IDs

```python
# In logging_utils.py
import contextvars

correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')

class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id.get() or 'none'
        return True
```

#### 10.2 Performance Logging

```python
# Add decorator for timing
import functools
import time

def log_timing(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                logger.debug(
                    f"{func.__name__} completed in {elapsed:.2f}s",
                    extra={'function': func.__name__, 'elapsed': elapsed}
                )
        return wrapper
    return decorator
```

**Estimated Effort:** 2-3 days
**Success Criteria:** Better observability, structured logs

---

### 11. Input Validation

**Problem:** Basic validation, unclear error messages

**Solution:**

#### 11.1 Consider Pydantic for Config

```python
# config.py
from pydantic import BaseModel, Field, validator

class Settings(BaseModel):
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = Field(default="text", pattern="^(text|json)$")
    log_file: Optional[str] = None
    random_seed: int = Field(default=42, ge=0)
    default_locale: str = Field(default="en", min_length=2, max_length=5)

    @validator('log_file')
    def validate_log_file(cls, v):
        if v is not None:
            path = Path(v).parent
            if not path.exists():
                raise ValueError(f"Log file directory does not exist: {path}")
        return v

    class Config:
        env_prefix = "LEXICLASS_"
```

#### 11.2 Validation Decorators

```python
# utils/validation.py
from functools import wraps
from pathlib import Path

def validate_file_exists(param_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            file_path = kwargs.get(param_name)
            if file_path and not Path(file_path).exists():
                raise FileNotFoundError(f"{param_name} not found: {file_path}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

**Estimated Effort:** 3-4 days
**Success Criteria:** Better error messages, early validation

---

### 12. Performance & Monitoring

**Problem:** Limited observability and benchmarking

**Solution:**

#### 12.1 Benchmarking Suite

Create `benchmarks/`:
```
benchmarks/
├── __init__.py
├── bench_tokenization.py
├── bench_features.py
├── bench_index.py
└── bench_classifier.py
```

Example `bench_tokenization.py`:
```python
import time
from lexiclass.tokenization import ICUTokenizer

def benchmark_tokenization(num_docs=1000, avg_length=500):
    tokenizer = ICUTokenizer()
    docs = [" ".join(["word"] * avg_length) for _ in range(num_docs)]

    start = time.perf_counter()
    for doc in docs:
        tokenizer.tokenize(doc)
    elapsed = time.perf_counter() - start

    throughput = num_docs / elapsed
    print(f"Tokenization: {throughput:.2f} docs/sec")
    return throughput

if __name__ == "__main__":
    benchmark_tokenization()
```

#### 12.2 Performance Regression Tests

```python
# tests/performance/test_benchmarks.py
import pytest

@pytest.mark.slow
def test_tokenization_performance():
    """Ensure tokenization doesn't regress."""
    throughput = benchmark_tokenization(num_docs=1000)
    assert throughput > 100, f"Tokenization too slow: {throughput} docs/sec"
```

**Estimated Effort:** 1 week
**Success Criteria:** Baseline benchmarks, regression detection

---

## 🟢 **LOW PRIORITY**

### 13. Code Organization

**Problem:** Flat structure may become hard to navigate

**Potential Future Structure:**
```
src/lexiclass/
├── __init__.py
├── __version__.py
├── core/
│   ├── __init__.py
│   ├── classifier.py
│   ├── index.py
│   └── features.py
├── tokenizers/
│   ├── __init__.py
│   ├── base.py
│   └── icu.py
├── encoders/
│   ├── __init__.py
│   └── label_encoders.py
├── io/
│   ├── __init__.py
│   ├── loaders.py
│   └── writers.py
├── utils/
│   ├── __init__.py
│   ├── logging.py
│   ├── memory.py
│   ├── config.py
│   └── validation.py
├── cli/
│   ├── __init__.py
│   └── main.py
├── datasets/
│   ├── __init__.py
│   └── wikipedia.py
├── exceptions.py
├── interfaces.py
└── plugins.py
```

**Note:** Only implement if codebase grows significantly (>5000 LOC)

---

### 14. Security

**Solution:**

#### 14.1 Security Scanning

Add to CI:
```yaml
- name: Run Bandit
  run: bandit -r src/ -f json -o bandit-report.json

- name: Check for vulnerabilities
  run: pip-audit
```

#### 14.2 SECURITY.md

```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities to: security@example.com

Do NOT create public GitHub issues for security vulnerabilities.
```

---

### 15. Advanced Features (Future)

- **Async I/O:** `async def` support for document loading
- **REST API:** FastAPI wrapper for HTTP endpoints
- **Docker:** Containerization for easy deployment
- **Distributed:** Ray/Dask for large-scale processing
- **Model Registry:** Centralized model versioning
- **Monitoring:** Prometheus/Grafana integration

---

### 16. Project Metadata Files

Create:
- `AUTHORS.md` - List of contributors
- `.editorconfig` - Consistent editor settings
- `CITATION.cff` - Academic citation format
- `.gitattributes` - Git LFS configuration if needed

---

## Specific Code Smells to Fix

### classifier.py

1. **Lines 39-73:** Remove commented-out `build_index()` method (dead code)
2. **Line 218:** Replace bare `except Exception` with specific exceptions
3. **Lines 28, 87, 153:** Long methods - consider extracting helper methods
4. **Line 177:** String conversion pattern repeated - extract to helper

### io.py

1. **Line 64:** Hacky `type('E', (), {...})` shim - create proper dataclass
2. **Lines 77-202:** `iter_documents_from_directory_parallel` is too complex - refactor
3. **Line 169:** Unreachable code after decorator definition

### index.py

1. **Long methods:** `build_index()` (80 lines) - extract phases
2. **Lines 83-113:** Complex nested function - extract to method
3. **Repeated patterns:** File opener logic repeated - DRY principle

### features.py

1. **Line 119:** Nested function `process_batch` - extract to method
2. **Lines 186-246:** Long `_filter_dictionary` method - break down
3. **Magic numbers:** BATCH_SIZE, PRUNE_AT - make configurable

### General

1. **Missing docstrings:** Add to all public methods
2. **Type hints:** Some missing return types
3. **Error messages:** Make more helpful with suggestions
4. **Logging:** Inconsistent logging levels

---

## Implementation Roadmap

### **Phase 1: Foundation (Weeks 1-2)**

**Goal:** Establish quality infrastructure

- [ ] Set up testing framework (pytest)
- [ ] Write unit tests for core modules (50% coverage)
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Add pre-commit hooks
- [ ] Configure mypy and fix type issues
- [ ] Create custom exception hierarchy
- [ ] Clean up code smells

**Deliverables:**
- Working test suite with 50%+ coverage
- Green CI pipeline
- Type-checked codebase
- Custom exceptions in use

---

### **Phase 2: Quality (Weeks 3-4)**

**Goal:** Improve code quality and documentation

- [ ] Increase test coverage to 70%+
- [ ] Add Ruff linting and formatting
- [ ] Complete API docstrings
- [ ] Set up MkDocs documentation
- [ ] Add examples and tutorials
- [ ] Create CONTRIBUTING.md
- [ ] Start CHANGELOG.md

**Deliverables:**
- 70%+ test coverage
- Complete API documentation
- Contributor guidelines
- Working documentation site

---

### **Phase 3: Polish (Weeks 5-6)**

**Goal:** Prepare for public release

- [ ] Improve packaging (enhanced pyproject.toml)
- [ ] Add badges to README
- [ ] Create benchmarking suite
- [ ] Set up dependency management (Poetry or requirements/)
- [ ] Add Dependabot
- [ ] Security scanning (Bandit, pip-audit)
- [ ] Increase test coverage to 85%+

**Deliverables:**
- PyPI-ready package
- Performance baselines
- Security scanning in CI
- 85%+ test coverage

---

### **Phase 4: Release (Week 7)**

**Goal:** First public release

- [ ] Final testing on multiple Python versions
- [ ] Prepare release notes
- [ ] Create git tag (v0.1.0)
- [ ] Publish to PyPI
- [ ] Publish documentation
- [ ] Announce release

**Deliverables:**
- v0.1.0 on PyPI
- Published documentation
- GitHub release with notes

---

### **Phase 5: Advanced (Ongoing)**

**Goal:** Continuous improvement

- [ ] Monitor usage and gather feedback
- [ ] Performance optimizations
- [ ] Additional features (async, REST API, etc.)
- [ ] Community building
- [ ] Regular dependency updates
- [ ] Expand test coverage to 90%+

---

## Success Metrics

### Code Quality
- **Test Coverage:** 0% → 50% → 70% → 85% → 90%
- **Type Coverage:** Unknown → 90%+ (mypy)
- **Documentation Coverage:** 40% → 90%+
- **Linter Score:** N/A → Pass with 0 errors

### Automation
- **CI Pass Rate:** N/A → 95%+
- **Build Time:** N/A → <5 minutes
- **Pre-commit Adoption:** 0% → 100% of contributors

### Project Health
- **Issue Response Time:** N/A → <48 hours
- **PR Review Time:** N/A → <72 hours
- **Security Vulnerabilities:** Unknown → 0 critical/high
- **Dependency Freshness:** Unknown → <6 months old

### Community
- **PyPI Downloads:** 0 → Track growth
- **GitHub Stars:** Current → Target +50 in 3 months
- **Contributors:** 1 → Target 5 in 6 months
- **Documentation Page Views:** 0 → Track

---

## Resource Requirements

### Time Investment
- **Phase 1 (Foundation):** ~40 hours
- **Phase 2 (Quality):** ~40 hours
- **Phase 3 (Polish):** ~30 hours
- **Phase 4 (Release):** ~10 hours
- **Total:** ~120 hours (3 months part-time)

### Tools & Services (Free Tier Sufficient)
- GitHub Actions (CI/CD)
- Codecov (Coverage reporting)
- ReadTheDocs or GitHub Pages (Documentation)
- PyPI (Package hosting)
- Dependabot (Dependency updates)

### Skills Needed
- Python packaging and distribution
- Testing with pytest
- CI/CD setup (GitHub Actions)
- Documentation writing (Markdown, MkDocs)
- Type systems (mypy)

---

## Risk Assessment

### High Risk
- **Breaking Changes:** Refactoring may break existing workflows
  - *Mitigation:* Semantic versioning, deprecation warnings, migration guide

### Medium Risk
- **Test Coverage:** Achieving 85%+ coverage is time-consuming
  - *Mitigation:* Incremental approach, focus on critical paths first

- **Documentation:** Comprehensive docs require significant effort
  - *Mitigation:* Start with critical user flows, expand iteratively

### Low Risk
- **CI/CD:** Well-established patterns exist
  - *Mitigation:* Use proven GitHub Actions workflows

- **Type Checking:** Gradual typing approach minimizes disruption
  - *Mitigation:* Start lenient, increase strictness over time

---

## Conclusion

This improvement plan provides a structured path to transform LexiClass from a functional project to a professional, production-ready library. The phased approach allows for incremental improvements while maintaining project functionality.

**Priority Order:**
1. **Testing** - Critical for confidence in changes
2. **CI/CD** - Automates quality checks
3. **Type Checking** - Catches errors early
4. **Documentation** - Enables user adoption
5. **Packaging** - Enables distribution
6. **Everything Else** - Nice-to-have improvements

**Next Steps:**
1. Review and prioritize specific items
2. Set up project board (GitHub Projects) to track progress
3. Begin Phase 1 with testing infrastructure
4. Regular progress reviews every 2 weeks

This plan is a living document and should be updated as priorities shift or new requirements emerge.
