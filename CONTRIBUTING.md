# Contributing

Thank you for helping improve AnchorPT.

## Report an issue

Use [GitHub Issues](https://github.com/yangongjiyu805-arch/anchorpt-multiomics/issues) and include:

- Repository commit and entry point.
- Operating system, Python, package versions, and GPU/driver information.
- Effective parameter JSON, evaluation arguments, and the relevant traceback.
- Input shape, modality labels, layer names, and feature-mask structure.
- Expected behavior and a minimal way to reproduce the problem.

For score discrepancies, include split/annotation versions, seeds, embedding key, and the complete metric inventory. See [reproducibility](docs/reproducibility.md).

## Propose a change

Keep changes focused. Explain which behavior changes, why it is needed, and how it was checked. Changes to preprocessing, transport objectives, training defaults, or evaluation aggregation should identify their effect on experiment comparability.

Keep documentation commands aligned with the actual entry points. Do not commit datasets, checkpoints, credentials, or generated results. Any synthetic example should be identified as synthetic and should not be presented as benchmark evidence.
