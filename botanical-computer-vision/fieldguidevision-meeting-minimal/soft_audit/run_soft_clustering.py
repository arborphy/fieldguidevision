"""Audit and redo soft clustering using frozen train-only preprocessing/models.
Run from any directory. EfficientNet only: other raw vectors are absent.
No new test evaluation. Seed variation is not sampling uncertainty.
"""
from pathlib import Path
import hashlib,itertools,json,warnings,platform
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score,roc_auc_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;BCV=HERE.parents[1]
KS=[5,10,15,20,30,50];SEEDS=[42,43,44,45,46]
def entropy(p):
    return -(p*np.log(np.maximum(p,1e-300))).sum(1)
def fit_gmm(z,k,cov,seed,n_init=3):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning)
        m=GaussianMixture(k,covariance_type=cov,reg_covar=1e-6,random_state=seed,n_init=n_init,max_iter=300).fit(z)
    return m,[str(w.message) for w in ws]
def bootstrap_auc(y,score,groups,n=500):
    # Resample entire individual plants, retaining their correlated images.
    rng=np.random.default_rng(42);unique=np.unique(groups);by={g:np.flatnonzero(groups==g) for g in unique};a=[]
    for _ in range(n):
        ix=np.concatenate([by[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ix]))==2:a.append(roc_auc_score(y[ix],score[ix]))
    return [float(x) for x in np.quantile(a,[.025,.975])]
def main():
    src=BCV/'data/embeddings/efficientnet_b0_strict/embeddings.npz'
    digest=hashlib.sha256(src.read_bytes()).hexdigest()
    assert digest=='b1b6e4f21f8b6ceb3e08f90ac8766e63a9dcce0caa82857068e85dc71679730b'
    d=np.load(src,allow_pickle=True);X=d['embeddings'].astype(float);ids=d['image_ids'].astype(str)
    man=pd.read_csv(BCV/'organ_benchmark/split_manifests/individual_disjoint.csv').set_index('image_id').loc[ids].reset_index()
    assert (man.split.values==d['splits'].astype(str)).all()
    tr=man.split.eq('train').values;va=man.split.eq('validation').values
    assert not set(man.individual_id[tr])&set(man.individual_id[va])
    organ=man.organ_category.replace({'whole tree (or vine)':'whole tree','whole plant':'whole tree','stem':'other','unspecified':'other','seed':'other'}).values
    scale=pd.read_csv(BCV/'scale_conditioned_analysis/data/image_visual_groups.csv').drop_duplicates('image_id').set_index('image_id').reindex(ids).shot_scale.fillna('uncertain').values
    audit=pd.read_csv(BCV/'organ_benchmark/ambiguity_audit/multi_label_audit.csv')
    filled=audit[[c for c in audit if c.startswith('visible_')]+['ambiguous','acceptable_labels']].notna().sum().to_dict()
    assert sum(filled.values())==0  # Existing CSV is an unfilled annotation template, not ground truth.
    # Reproduce the original all-corpus PCA32/diag20 setting; no accuracy claim.
    oldz=PCA(32,random_state=42).fit_transform(X)
    oldp=[]
    for seed in SEEDS:
        m,_=fit_gmm(oldz,20,'diag',seed,n_init=1);oldp.append(m.predict_proba(oldz))
    oldh=[entropy(p) for p in oldp]
    old=dict(rerun_note='Same configuration, current sklearn 1.8.0/float64; archived report names sklearn 1.9 and float32 inputs. Not an exact numeric reproduction.',frac_pmax_lt06=float((oldp[0].max(1)<.6).mean()),
             mean_seed_ARI=float(np.mean([adjusted_rand_score(a.argmax(1),b.argmax(1)) for a,b in itertools.combinations(oldp,2)])),
             mean_entropy_rank_rho=float(np.mean([spearmanr(a,b).statistic for a,b in itertools.combinations(oldh,2)])))
    # The original 'correct' field belongs to species, not organ, prediction.
    pred=pd.read_csv(BCV/'data/embeddings/efficientnet_b0_strict/all_predictions.csv')
    assert 'predicted_species' in pred and 'ground_truth_species' in pred
    old['original_error_target']='species classification (all_predictions.csv), not organ classification'
    print('reproduced old',json.dumps(old),flush=True)
    # Fixed supervised organ reference; validation errors independently defined.
    known=organ!='other'
    probe=LogisticRegression(C=10,class_weight='balanced',max_iter=2000,random_state=42).fit(X[tr&known],organ[tr&known])
    opred=probe.predict(X[va]);wrong=opred!=organ[va];valid_org=known[va]
    probe_record=dict(model='fixed C=10 balanced multinomial logistic regression on raw embeddings; train fit only',
                      n_train=int((tr&known).sum()),n_val=int(valid_org.sum()),n_val_wrong=int(wrong[valid_org].sum()),
                      val_macroF1=float(f1_score(organ[va][valid_org],opred[valid_org],average='macro',zero_division=0)))
    runs=[];stable=[];models={};probs={};zs={};pca_info={}
    # Main fixed-dimensional comparison and separate PCA dimension sensitivity.
    configs=[(32,k,cov,SEEDS) for k in KS for cov in ['diag','tied','full']]
    configs += [(dim,k,cov,SEEDS[:3]) for dim in [8,16] for k in [5,10,20,50] for cov in ['diag','tied']]
    for dim in [8,16,32]:
        p=PCA(dim,whiten=False,random_state=42).fit(X[tr]);zs[dim]=(p.transform(X[tr]),p.transform(X[va]))
        pca_info[dim]=float(p.explained_variance_ratio_.sum())
    for dim,k,cov,seeds in configs:
        zt,zv=zs[dim];pp=[];hh=[];top=[]
        for seed in seeds:
            m,ws=fit_gmm(zt,k,cov,seed)
            p=m.predict_proba(zv);h=entropy(p)/np.log(k)
            assert np.isfinite(p).all() and np.allclose(p.sum(1),1) and (p>=0).all()
            assert np.isfinite(m.score_samples(zv)).all() and ((h>=-1e-10)&(h<=1+1e-10)).all()
            pp.append(p);hh.append(h);top.append(p.argmax(1))
            effective=m.predict_proba(zt).sum(0)
            row=dict(dim=dim,k=k,covariance=cov,seed=seed,converged=bool(m.converged_),n_iter=int(m.n_iter_),
                     train_BIC=float(m.bic(zt)),val_loglik=float(m.score(zv)),val_mean_entropy=float(h.mean()),
                     val_pmax_lt06=float((p.max(1)<.6).mean()),val_pmax_lt08=float((p.max(1)<.8).mean()),
                     val_pmax_lt09=float((p.max(1)<.9).mean()),val_mean_effective_components=float(np.exp(entropy(p)).mean()),
                     min_train_effective_membership=float(effective.min()),train_components_effective_lt10=int((effective<10).sum()),
                     val_organ_error_AUC=float(roc_auc_score(wrong[valid_org],h[valid_org])),
                     val_midrange_AUC=float(roc_auc_score((scale[va]=='mid-range')[scale[va]!='uncertain'],h[scale[va]!='uncertain'])),
                     warnings=' | '.join(ws))
            runs.append(row)
            if dim==32:models[(k,cov,seed)]=m;probs[(k,cov,seed)]=p
        pair_ari=[adjusted_rand_score(a,b) for a,b in itertools.combinations(top,2)]
        pair_h=[spearmanr(a,b).statistic for a,b in itertools.combinations(hh,2)]
        sets=[set(np.argsort(h)[-25:]) for h in hh]
        jac=[len(a&b)/len(a|b) for a,b in itertools.combinations(sets,2)]
        stable.append(dict(dim=dim,k=k,covariance=cov,seed_ARI=np.mean(pair_ari),seed_entropy_rank_rho=np.nanmean(pair_h),
                           seed_top25_entropy_Jaccard=np.mean(jac)))
        print(f'done dim={dim} K={k} {cov}',flush=True)
    r=pd.DataFrame(runs);r.to_csv(HERE/'runs.csv',index=False)
    keys=['dim','k','covariance'];metrics=[c for c in r if c not in keys+['seed','warnings','converged']]
    s=r.groupby(keys)[metrics].agg(['mean','std']).reset_index();s.columns=['_'.join(c).rstrip('_') for c in s.columns]
    s=s.merge(pd.DataFrame(stable),on=keys);s.to_csv(HERE/'summary.csv',index=False)
    # Select only within fixed 32-D space using training BIC, excluding non-convergence.
    best=r[(r.dim==32)&r.converged].sort_values('train_BIC').iloc[0]
    k=int(best.k);cov=best.covariance;seed=int(best.seed);m=models[(k,cov,seed)];p=probs[(k,cov,seed)]
    h=entropy(p)/np.log(k);nll=-m.score_samples(zs[32][1]);groups=man.individual_id.values[va]
    diagnostic=dict(selected_by='minimum train BIC among converged 32-D candidates; no validation/test tuning',
                    selected=best.to_dict(),organ_reference_probe=probe_record,
                    val_organ_error_AUC_CI95=bootstrap_auc(wrong[valid_org],h[valid_org],groups[valid_org]),
                    val_midrange_AUC_CI95=bootstrap_auc((scale[va]=='mid-range')[scale[va]!='uncertain'],h[scale[va]!='uncertain'],groups[scale[va]!='uncertain']),
                    old_setting_reproduced=old,multilabel_audit_filled_cells=filled,pca_train_variance=pca_info,
                    n_train=int(sum(tr)),n_val=int(sum(va)),test_used=False,input_sha256=digest,
                    sklearn=sklearn.__version__,python=platform.python_version())
    # Soft organ composition derives from train only; component IDs are not organ labels.
    ptr=m.predict_proba(zs[32][0]);classes=np.unique(organ[tr]);composition=np.stack([ptr[organ[tr]==c].sum(0) for c in classes],axis=1)
    composition/=composition.sum(1,keepdims=True)
    pd.DataFrame(composition,columns=classes).rename_axis('component').to_csv(HERE/'train_component_organ_composition.csv')
    out=man[va][['image_id','individual_id','species','organ_category','thumbnail_url']].copy()
    out['organ_prediction']=opred;out['organ_wrong']=wrong;out['known_organ']=valid_org
    out['pmax']=p.max(1);out['entropy_normalized']=h;out['negative_log_density']=nll
    order=np.argsort(p,axis=1)
    out['top1_component']=order[:,-1];out['top2_component']=order[:,-2]
    out['p2']=np.take_along_axis(p,order[:,-2,None],axis=1).ravel()
    config_p=[probs[(k,cov,se)] for se in SEEDS]
    hs=np.stack([entropy(a)/np.log(k) for a in config_p]);out['mean_seed_entropy']=hs.mean(0);out['sd_seed_entropy']=hs.std(0,ddof=1)
    out['top1_component_organ']=classes[composition[out.top1_component].argmax(1)]
    out['top2_component_organ']=classes[composition[out.top2_component].argmax(1)]
    out.to_csv(HERE/'validation_memberships.csv',index=False)
    out.sort_values('mean_seed_entropy',ascending=False).head(12).to_csv(HERE/'ambiguous_validation_cases.csv',index=False)
    np.savez_compressed(HERE/'selected_responsibilities.npz',image_ids=ids[va],probabilities=p,
                        train_component_organ_probabilities=composition,organ_classes=classes)
    (HERE/'audit.json').write_text(json.dumps(diagnostic,indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o)))
    fig,ax=plt.subplots(2,2,figsize=(11,7))
    for cov in ['diag','tied','full']:
        q=s[(s.dim==32)&s.covariance.eq(cov)]
        for axes,col in [(ax[0,0],'val_loglik'),(ax[0,1],'val_pmax_lt06'),(ax[1,0],'val_mean_entropy')]:
            axes.errorbar(q.k,q[col+'_mean'],yerr=q[col+'_std'],marker='o',capsize=3,label=cov)
    ax[0,0].set(xlabel='K',ylabel='Mean held-out log density',title='Same train PCA-32 space (higher is better)')
    ax[0,1].set(xlabel='K',ylabel='Validation fraction with pmax < 0.6',title='Softness depends on covariance and K',ylim=(0,.12))
    ax[1,0].set(xlabel='K',ylabel='Mean H / log(K)',title='Normalized membership entropy',ylim=(0,.12))
    for axes in [ax[0,0],ax[0,1],ax[1,0]]:axes.legend(fontsize=8)
    ax[1,1].scatter(h[(~wrong)&valid_org],nll[(~wrong)&valid_org],label='Organ probe correct',alpha=.6,s=14)
    ax[1,1].scatter(h[wrong&valid_org],nll[wrong&valid_org],label='Organ probe wrong',alpha=.7,s=14)
    ax[1,1].set(xlabel='H / log(K)',ylabel='Negative log density (may be negative)',title=f'Train-BIC choice: K={k}, {cov}, seed {seed}')
    ax[1,1].legend(fontsize=8)
    fig.suptitle('Soft clustering audit: EfficientNet; train-only fits, validation diagnostics',fontsize=12)
    fig.tight_layout();fig.savefig(HERE/'soft_diagnostics.png',dpi=140);plt.close(fig)
    print(json.dumps(diagnostic,default=str),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
