# Preprocessing

| Script | Input |
| :--- | :--- |
| [prepare_gxr_gtf_rna_atac.py](prepare_gxr_gtf_rna_atac.py) | Matrix Market RNA–ATAC; GTF or gene–region mapping |
| [prepare_h5ad_gtf_rna_atac.py](prepare_h5ad_gtf_rna_atac.py) | AnnData RNA–ATAC; GTF |
| [prepare_h5ad_hgnc_rna_protein.py](prepare_h5ad_hgnc_rna_protein.py) | AnnData RNA–protein; HGNC JSON |
| [prepare_tripbmc_unpaired_split.py](prepare_tripbmc_unpaired_split.py) | Dataset-specific three-modality splits |

Use `--help` for arguments. Keep the scripts together: the AnnData/GTF route imports its companion module.

[Input formats and commands](../../docs/preprocessing.md)
