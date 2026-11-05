# Data preprocessing

[Code overview](../README.md) · [Full input guide](../../docs/preprocessing.md)

| Script | Input / task |
| :--- | :--- |
| [prepare_gxr_gtf_rna_atac.py](prepare_gxr_gtf_rna_atac.py) | RNA and ATAC Matrix Market directories; GTF or gene–region mapping |
| [prepare_h5ad_gtf_rna_atac.py](prepare_h5ad_gtf_rna_atac.py) | RNA and ATAC AnnData files; GTF annotation |
| [prepare_h5ad_hgnc_rna_protein.py](prepare_h5ad_hgnc_rna_protein.py) | RNA and protein AnnData files; HGNC-based feature mapping |
| [prepare_tripbmc_unpaired_split.py](prepare_tripbmc_unpaired_split.py) | Dataset-specific disjoint RNA/ATAC/protein split construction |

The AnnData/GTF script imports the companion Matrix Market preprocessor from this directory. Keep these files together.

Run from the repository root:

```bash
python code/preprocessing/prepare_gxr_gtf_rna_atac.py --help
```

Datasets and annotations are supplied separately. See the [preprocessing guide](../../docs/preprocessing.md) for input schemas and complete commands.
