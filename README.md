# VISTA

Annotation-consistent cross-sensor spectral harmonization for world-model inputs.

VISTA aligns PRISMA spectra to the DESIS reference scale and refines the spectra while preserving the class assigned by the fixed classifier after global harmonization.

## Reproduce the reported results

The [code package](code/README.md) includes frozen models, public USGS input tables, saved experimental results, and a unified verification command. Use Python 3.12:

```bash
cd code
python -m venv .venv
# Activate .venv, then:
python -m pip install -r requirements-verify.txt
python check_manifest.py
python prepare_inputs.py --out inputs
python verify_reproduction.py --data-dir inputs --out verified
```

This replays saved models and checks the reported numerical results. It does not train models or select parameters. The verified package covers 16,581 held-out spectra in the principal five-fold study, the controlled comparison, and the preserved supplementary studies. See [reproduction coverage and requirements](code/REPRODUCIBILITY.md) and the [executed verification report](code/verification/verification_report.json).

Use [the inference instructions](code/README.md#frozen-model-inference) to apply the saved models to PRISMA measurements. Original fitting scripts remain available separately.

## Files and history

The current manuscript results are under `code/reported_results/converged_rerun/`. Earlier `current_common_utility/` and `current_reference_anchor/` directories are retained as historical records; they are not the current manuscript results. The unified verifier follows the explicitly named current and supplementary records in `code/MANIFEST.json`.

## License and data

The code is licensed under [MIT](LICENSE). USGS input data retain their public-domain status and source attribution; see [data sources](code/data/README.md) and [third-party attribution](code/THIRD_PARTY.md).
