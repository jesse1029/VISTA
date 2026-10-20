# Dependencies and provenance

VISTA's Python source is distributed under the root MIT license.
`vendor/original_common_utility.py` is the project's archived experimental
implementation, retained to preserve its original algorithms and parameter
definitions. It is not a copied distribution of a third-party library.

Dependencies are installed separately through pip and keep their own licenses:

| Dependency | License |
|---|---|
| NumPy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |

The original research programs and outputs in `supplement/records` are kept
unchanged for source-hash verification. Their historical filenames and report
wording are provenance records. The public input data and their terms are
described in [data/README.md](data/README.md).
