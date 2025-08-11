#!/usr/bin/env python3
"""
Compatibility wrapper for the legacy monolithic script.

The functionality now lives in the `lexiclass` package with a Typer CLI.
Install the project and use:

    lexiclass --help

Or import programmatically:

    from lexiclass import SVMDocumentClassifier, DocumentIndex, FeatureExtractor, ICUTokenizer

To run the Wikipedia demo from CLI:

    lexiclass demo-wikipedia
"""

from lexiclass import (
    ICUTokenizer,
    FeatureExtractor,
    BinaryClassEncoder,
    MultiClassEncoder,
    DocumentLoader,
    load_labels,
    DocumentIndex,
    SVMDocumentClassifier,
)

def main():  # pragma: no cover - wrapper only
    try:
        from lexiclass.cli.main import app
        # Invoke the CLI when executed directly
        app()
    except Exception as exc:  # noqa: BLE001
        import sys
        sys.stderr.write(f"Error launching CLI: {exc}\n")
        sys.stderr.write("Install the package and run: lexiclass --help\n")


if __name__ == "__main__":
    main()


