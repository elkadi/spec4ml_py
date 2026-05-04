# Spec4ML for Python

Spec4ML for Python is a package for handling and analyzing spectral data with special considerations for machine-learning applications.

[![CI](https://github.com/elkadi/spec4ml_py/actions/workflows/ci.yml/badge.svg)](https://github.com/elkadi/spec4ml_py/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/spec4ml-py.svg)](https://pypi.org/project/spec4ml-py/)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxx)

## Installation

```bash
pip install spec4ml-py
# or from GitHub
pip install git+https://github.com/elkadi/spec4ml_py.git
```

## Quickstart

```python
from spec4ml_py import __version__

print("Spec4ML for Python", __version__)
```

## Documentation

Complete package documentation is available here:

- [Full documentation](docs/README.md)

The full documentation covers:

- installation and development setup,
- data layout conventions,
- evaluation workflows,
- technical replicate handling patterns,
- prediction aggregation,
- regression metrics,
- feature-block importance,
- ensemble workflows,
- troubleshooting,
- relationship to the R package and Studio app.

## How to cite

See [CITATION.cff](CITATION.cff) or the DOI badge above.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Related repositories

- R package: `https://github.com/elkadi/spec4ml`
- Streamlit app: `https://github.com/elkadi/SpecML-Studio`
