"""Prepare feature-aligned RNA-protein inputs from QC h5ad files."""
import argparse, hashlib, json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# Optionally map detailed BMMC labels to broad cell classes.
BMMC_CELLTYPE = {
'CD4+ T naive':'CD4 T','CD4+ T activated':'CD4 T','CD4+ T activated integrinB7+':'CD4 T','CD4+ T CD314+ CD45RA+':'CD4 T','T reg':'CD4 T',
'CD8+ T naive':'CD8 T','CD8+ T':'CD8 T','CD8+ T naive CD127+ CD26- CD101-':'CD8 T','CD8+ T CD49f+':'CD8 T','CD8+ T TIGIT+ CD45RO+':'CD8 T','CD8+ T CD57+ CD45RA+':'CD8 T','CD8+ T CD69+ CD45RO+':'CD8 T','CD8+ T TIGIT+ CD45RA+':'CD8 T','CD8+ T CD69+ CD45RA+':'CD8 T','CD8+ T CD57+ CD45RO+':'CD8 T','MAIT':'CD8 T','gdT TCRVD2+':'CD8 T','gdT CD158b+':'CD8 T','dnT':'CD8 T',
'Naive CD20+ B IGKC+':'B cell','Naive CD20+ B IGKC-':'B cell','Naive CD20+ B':'B cell','B1 B IGKC+':'B cell','B1 B IGKC-':'B cell','B1 B':'B cell','Transitional B':'B cell',
'Plasmablast IGKC+':'Plasma cell','Plasmablast IGKC-':'Plasma cell','Plasma cell IGKC+':'Plasma cell','Plasma cell IGKC-':'Plasma cell','Plasma cell':'Plasma cell',
'NK':'NK','NK CD158e1+':'NK','CD14+ Mono':'Mono','CD16+ Mono':'Mono','pDC':'DC','cDC1':'DC','cDC2':'DC','ILC':'ILC','ILC1':'ILC',
'HSC':'Progenitor','Lymph prog':'Progenitor','G/M prog':'Progenitor','MK/E prog':'Progenitor','ID2-hi myeloid prog':'Progenitor','T prog cycling':'Progenitor',
'Erythroblast':'Erythroid','Normoblast':'Erythroid','Proerythroblast':'Erythroid','Reticulocyte':'Erythroid'}

def args_parser():
 p=argparse.ArgumentParser()
 p.add_argument('--rna-h5ad',required=True);p.add_argument('--protein-h5ad',required=True);p.add_argument('--hgnc-json',required=True);p.add_argument('--output',required=True)
 p.add_argument('--mapping-report',default='');p.add_argument('--rna-counts-layer',default='counts');p.add_argument('--protein-counts-layer',default='counts')
 p.add_argument('--batch-key',default='batch');p.add_argument('--celltype-key',default='cell_type');p.add_argument('--rna-dataset-label',default='');p.add_argument('--protein-dataset-label',default='')
 p.add_argument('--top-genes',type=int,default=5000);p.add_argument('--target-sum',type=float,default=1e4);p.add_argument('--hvg-flavor',default='seurat')
 p.add_argument('--harmonize-bmmc-celltypes',action='store_true');p.add_argument('--ambiguous-map-policy',choices=('first','keep','error'),default='first')
 return p.parse_args()

def counts_layer(a,layer,label):
 if layer not in a.layers: raise KeyError(f'{label} missing layer {layer!r}')
 x=a.layers[layer];x=x.tocsr().astype(np.float32) if sparse.issparse(x) else sparse.csr_matrix(x,dtype=np.float32)
 if x.shape!=a.shape: raise ValueError(f'{label} counts shape mismatch')
 if x.data.size and (np.any(x.data<0) or not np.allclose(x.data,np.rint(x.data),atol=1e-6)): raise ValueError(f'{label} counts are not integer-like')
 return x

def make_obs(a,modality,batch_key,celltype_key,dataset,harmonize):
 missing=[k for k in (batch_key,celltype_key) if k not in a.obs]
 if missing: raise KeyError(f'{modality} obs missing {missing}')
 raw=a.obs[celltype_key].astype(str)
 if harmonize:
  mapped=raw.map(BMMC_CELLTYPE)
  if mapped.isna().any(): raise ValueError(f'{modality} unmapped cell types: {sorted(raw[mapped.isna()].unique())}')
  celltype=mapped.to_numpy(dtype=object)
 else: celltype=raw.to_numpy(dtype=object)
 original=a.obs_names.astype(str).to_numpy(dtype=object)
 obs=pd.DataFrame(index=pd.Index([f'{modality}_{x}' for x in original],name='cell'))
 obs['original_barcode']=pd.Categorical(original);obs['modality']=pd.Categorical([modality]*a.n_obs);obs['batch']=pd.Categorical(a.obs[batch_key].astype(str).to_numpy())
 obs['cell_type']=pd.Categorical(celltype);obs['original_cell_type']=pd.Categorical(raw.to_numpy());obs['dataset']=pd.Categorical([dataset]*a.n_obs)
 return obs

def hgnc_docs(path):
 obj=json.loads(path.read_text(encoding='utf-8'));docs=obj.get('response',{}).get('docs',[]) if isinstance(obj,dict) else obj
 docs=[d for d in docs if d.get('status','Approved')=='Approved']
 if not docs: raise ValueError('HGNC JSON contains no approved records')
 return docs

# Resolve approved, previous, and alias HGNC symbols for each marker.
def map_markers(markers,docs,policy):
 mapped=[];rows=[]
 for marker in markers:
  cand=[]
  for d in docs:
   kinds=[]
   if marker==str(d.get('symbol','')): kinds.append('approved')
   if marker in [str(x) for x in d.get('prev_symbol',[]) or []]: kinds.append('previous')
   if marker in [str(x) for x in d.get('alias_symbol',[]) or []]: kinds.append('alias')
   if kinds: cand.append((str(d['symbol']),'+'.join(kinds),str(d.get('hgnc_id',''))))
  if len(cand)==1: gene,kind,status=cand[0][0],cand[0][1],'mapped'
  elif not cand: gene,kind,status=marker,'none','kept_unmapped'
  elif policy=='error': raise ValueError(f'Ambiguous HGNC mapping {marker}: {[x[0] for x in cand]}')
  elif policy=='keep': gene,kind,status=marker,'ambiguous','kept_ambiguous'
  else: gene,kind,status=cand[0][0],cand[0][1],'mapped_ambiguous_first'
  mapped.append(gene);rows.append({'original_protein':marker,'mapped_gene':gene,'status':status,'match_type':kind,'candidate_count':len(cand),'candidate_symbols':'|'.join(x[0] for x in cand),'candidate_hgnc_ids':'|'.join(x[2] for x in cand)})
 dup=pd.Series(mapped)[pd.Series(mapped).duplicated(keep=False)].unique().tolist()
 if dup: raise ValueError(f'HGNC mapping creates duplicate features: {dup}')
 return mapped,pd.DataFrame(rows)

def protein_clr(x):
 # Apply sparse CLR normalization to ADT counts.
 out=x.tocsc().astype(np.float32,copy=True);col=np.repeat(np.arange(out.shape[1]),np.diff(out.indptr));sums=np.bincount(col,weights=np.log1p(out.data),minlength=out.shape[1]);den=np.exp(sums/out.shape[0]);out.data=np.log1p(out.data/den[col]).astype(np.float32);return out.tocsr()

def rna_process(counts,obs,genes,top,target,flavor):
 work=ad.AnnData(X=counts.copy(),obs=obs.copy(),var=pd.DataFrame(index=pd.Index(genes)))
 sc.pp.normalize_total(work,target_sum=target);sc.pp.log1p(work)
 if 0<top<work.n_vars:
  kw={'n_top_genes':top,'flavor':flavor}
  if work.obs['batch'].nunique()>1: kw['batch_key']='batch'
  sc.pp.highly_variable_genes(work,**kw);hvgs=work.var_names[work.var['highly_variable']].astype(str).tolist()
 else: hvgs=work.var_names.astype(str).tolist()
 return work.X.tocsr().astype(np.float32),hvgs

# Project a modality matrix into the shared feature index.
def align(x,source,target):
 lookup={n:i for i,n in enumerate(source)};src=[];dst=[]
 for j,n in enumerate(target):
  if n in lookup: src.append(lookup[n]);dst.append(j)
 if not src: return sparse.csr_matrix((x.shape[0],len(target)),dtype=np.float32)
 coo=x[:,src].tocoo();cols=np.asarray(dst,dtype=np.int64)[coo.col]
 return sparse.csr_matrix((coo.data,(coo.row,cols)),shape=(x.shape[0],len(target)),dtype=np.float32)

def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()

# Build the feature union and modality-specific observation masks.
def main():
 a=args_parser();rp,pp,hp,out=Path(a.rna_h5ad),Path(a.protein_h5ad),Path(a.hgnc_json),Path(a.output);report=Path(a.mapping_report) if a.mapping_report else out.with_suffix('.protein_hgnc_mapping.tsv')
 print(f'[rna-protein] RNA {rp}',flush=True);rna=ad.read_h5ad(rp);print(f'[rna-protein] Protein {pp}',flush=True);prot=ad.read_h5ad(pp)
 rc=counts_layer(rna,a.rna_counts_layer,'RNA');pc=counts_layer(prot,a.protein_counts_layer,'Protein');genes=rna.var_names.astype(str).tolist();markers=prot.var_names.astype(str).tolist()
 if len(set(genes))!=len(genes) or len(set(markers))!=len(markers): raise ValueError('Input feature names must be unique')
 ro=make_obs(rna,'RNA',a.batch_key,a.celltype_key,a.rna_dataset_label or rp.stem,a.harmonize_bmmc_celltypes);po=make_obs(prot,'Protein',a.batch_key,a.celltype_key,a.protein_dataset_label or pp.stem,a.harmonize_bmmc_celltypes)
 rt,hvgs=rna_process(rc,ro,genes,a.top_genes,a.target_sum,a.hvg_flavor);pt=protein_clr(pc);mapped,mapping=map_markers(markers,hgnc_docs(hp),a.ambiguous_map_policy);report.parent.mkdir(parents=True,exist_ok=True);mapping.to_csv(report,sep='\t',index=False)
 features=list(hvgs);seen=set(features)
 for g in mapped:
  if g not in seen: features.append(g);seen.add(g)
 counts=sparse.vstack([align(rc,genes,features),align(pc,mapped,features)],format='csr').astype(np.float32);train=sparse.vstack([align(rt,genes,features),align(pt,mapped,features)],format='csr').astype(np.float32);obs=pd.concat([ro,po])
 rs,ps=set(genes),set(mapped);rm=np.asarray([x in rs for x in features]);pm=np.asarray([x in ps for x in features]);lookup=dict(zip(mapped,markers));var=pd.DataFrame(index=pd.Index(features,name='gene'));var['rna_observed']=rm;var['protein_observed']=pm;var['highly_variable']=[x in set(hvgs) for x in features];var['feature_origin']=pd.Categorical(['both' if r and p else 'RNA' if r else 'Protein' for r,p in zip(rm,pm)]);var['original_protein_marker']=[lookup.get(x,'') for x in features]
 data=ad.AnnData(X=train.copy(),obs=obs,var=var);data.layers['counts']=counts;data.layers['train_input']=train;data.uns['feature_list']={'RNA':np.flatnonzero(rm).astype(np.int64),'Protein':np.flatnonzero(pm).astype(np.int64)};data.uns['rna_hvg']=np.asarray(hvgs,dtype=str);data.uns['prot_hvg']=np.asarray(mapped,dtype=str);status=mapping['status'].value_counts().to_dict();data.uns['preprocess']={'source':'anchorpt_hgnc_rna_protein_scmrdr','rna_h5ad':str(rp),'protein_h5ad':str(pp),'rna_counts_layer':a.rna_counts_layer,'protein_counts_layer':a.protein_counts_layer,'target_sum':float(a.target_sum),'hvg_flavor':a.hvg_flavor,'top_genes':int(a.top_genes),'selected_rna_hvgs':len(hvgs),'protein_markers':len(markers),'union_features':len(features),'protein_normalization':'scMRDR_muon_style_clr','hgnc_json':str(hp),'hgnc_json_sha256':file_sha(hp),'hgnc_mapping_report':str(report),'hgnc_mapping_status':status,'ambiguous_map_policy':a.ambiguous_map_policy,'harmonize_bmmc_celltypes':bool(a.harmonize_bmmc_celltypes)}
 out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists(): raise FileExistsError(f'Refusing overwrite: {out}')
 data.write_h5ad(out);print(f'[rna-protein] wrote {out} shape={data.shape}',flush=True);print(f'[rna-protein] mapping {status}',flush=True);print(f'[rna-protein] observed RNA={rm.sum()} Protein={pm.sum()} overlap={(rm&pm).sum()}',flush=True)
if __name__=='__main__': main()


