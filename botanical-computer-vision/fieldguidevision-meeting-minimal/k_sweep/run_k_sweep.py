"""Reproduce K sensitivity from checked-in vectors. Run from any directory.
KMeans: raw L2-normalized EfficientNet vectors; fit train, freeze and assign val.
Five seeds x ten initializations. No test use; no automatic K selection.
NSC approximates each seed-42 partition using PCA-64 fit separately in each
training-group CV fold. Teacher partition fixed from all train (descriptive
partition approximation, not independent biological-label validation).
"""
from pathlib import Path
import hashlib, itertools, json, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari, silhouette_score, f1_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
BCV=HERE.parents[1]
KS=[5,10,15,20,30,50]
SEEDS=[42,43,44,45,46]
GRID=np.r_[0,np.linspace(.25,8,32)]

class NSC:
    def fit(self,Z,y,delta):
        self.classes,nk=np.unique(y,return_counts=True)
        n=len(y); K=len(nk); mean=Z.mean(0)
        cents=np.stack([Z[y==k].mean(0) for k in self.classes])
        within=sum(((Z[y==k]-cents[i])**2).sum(0) for i,k in enumerate(self.classes))
        s=np.sqrt(within/(n-K)); self.sd=np.maximum(s+np.median(s),1e-12)
        m=np.sqrt(1/nk-1/n)[:,None]
        d=(cents-mean)/(m*self.sd)
        shr=np.sign(d)*np.maximum(abs(d)-delta,0)
        self.active=shr!=0; self.cents=mean+m*self.sd*shr
        self.prior=np.log(nk/n)
        return self
    def predict(self,Z):
        # Drop the class-independent squared-norm term.
        centers=self.cents/self.sd
        scores=-2*(Z/self.sd)@centers.T+(centers**2).sum(1)-2*self.prior
        return self.classes[scores.argmin(1)]


def main():
    npz=BCV/'data/embeddings/efficientnet_b0_strict/embeddings.npz'
    digest=hashlib.sha256(npz.read_bytes()).hexdigest()
    assert digest=='b1b6e4f21f8b6ceb3e08f90ac8766e63a9dcce0caa82857068e85dc71679730b'
    d=np.load(npz,allow_pickle=True); X=d['embeddings'].astype(float); ids=d['image_ids'].astype(str)
    man=pd.read_csv(BCV/'organ_benchmark/split_manifests/individual_disjoint.csv').set_index('image_id').loc[ids].reset_index()
    assert len(ids)==1641 and (man.split.values==d['splits'].astype(str)).all()
    assert np.allclose(np.linalg.norm(X,axis=1),1,atol=1e-5)
    tr=man.split.eq('train').values; va=man.split.isin(['val','validation']).values
    assert not set(man.individual_id[tr]) & set(man.individual_id[va])
    scale=pd.read_csv(BCV/'scale_conditioned_analysis/data/image_visual_groups.csv').drop_duplicates('image_id').set_index('image_id').reindex(ids).shot_scale.fillna('uncertain').values
    organ=man.organ_category.replace({'whole tree (or vine)':'whole tree','whole plant':'whole tree','stem':'other','unspecified':'other','seed':'other'}).values
    pca=PCA(64,whiten=True,random_state=42).fit(X[tr]); Ztr=pca.transform(X[tr]); Zva=pca.transform(X[va])
    folds=[]
    for a,b in GroupKFold(5).split(X[tr],groups=man.individual_id.values[tr]):
        assert not set(man.individual_id.values[tr][a]) & set(man.individual_id.values[tr][b])
        p=PCA(64,whiten=True,random_state=42).fit(X[tr][a])
        folds.append((a,b,p.transform(X[tr][a]),p.transform(X[tr][b])))
    runs=[]; summary=[]; curves=[]; assignment=[]; sizes=[]
    for k in KS:
        predictions=[]; train42=None; val42=None
        for seed in SEEDS:
            km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(X[tr])
            labels=km.predict(X[va]); predictions.append(labels)
            counts=np.bincount(km.labels_,minlength=k)
            runs.append(dict(k=k,seed=seed,val_silhouette=silhouette_score(X[va],labels) if 1<len(set(labels))<sum(va) else np.nan,
                             val_organ_AMI=ami(organ[va],labels),val_scale_AMI=ami(scale[va],labels),val_species_AMI=ami(man.species.values[va],labels),
                             train_inertia_per_image=km.inertia_/sum(tr),train_min_cluster=int(counts.min()),train_clusters_lt10=int((counts<10).sum()),
                             val_clusters_observed=len(set(labels))))
            if seed==42:
                train42=km.labels_;val42=labels
                for split,mask,labs in [('train',tr,train42),('val',va,val42)]:
                    assignment.extend(dict(image_id=i,k=k,split=split,cluster_id=int(c)) for i,c in zip(ids[mask],labs))
                for cid in range(k):
                    mask=labels==cid; n=int(mask.sum())
                    o=pd.Series(organ[va][mask]).value_counts(); sp=pd.Series(man.species.values[va][mask]).value_counts()
                    sizes.append(dict(k=k,cluster_id=cid,train_n=int(counts[cid]),val_n=n,dominant_organ=o.index[0] if n else '',
                                      organ_share=o.iloc[0]/n if n else np.nan,top_species_share=sp.iloc[0]/n if n else np.nan))
        stability=[ari(a,b) for a,b in itertools.combinations(predictions,2)]
        df=pd.DataFrame([r for r in runs if r['k']==k])
        row=dict(k=k,seed_ARI_mean=float(np.mean(stability)),seed_ARI_min=float(min(stability)))
        for c in ['val_silhouette','val_organ_AMI','val_scale_AMI','val_species_AMI','train_inertia_per_image','train_clusters_lt10','val_clusters_observed']:
            row[c]=df[c].mean();row[c+'_sd']=df[c].std()
        labels_all=np.arange(k)
        for delta in GRID:
            f=[]
            for a,b,za,zb in folds:
                model=NSC().fit(za,train42[a],delta)
                f.append(f1_score(train42[b],model.predict(zb),labels=labels_all,average='macro',zero_division=0))
            model=NSC().fit(Ztr,train42,delta)
            curves.append(dict(k=k,delta=delta,cv_macroF1=np.mean(f),cv_se=np.std(f,ddof=1)/np.sqrt(5),
                               coords_any_class=int(model.active.any(0).sum()),coords_mean_per_class=model.active.sum(1).mean()))
        c=pd.DataFrame([r for r in curves if r['k']==k]); best=c.loc[c.cv_macroF1.idxmax()]
        selected=c[c.cv_macroF1>=best.cv_macroF1-best.cv_se].iloc[-1]
        chosen=NSC().fit(Ztr,train42,selected.delta)
        row.update(delta_1se=selected.delta,cv_F1_1se=selected.cv_macroF1,coords_1se=selected.coords_any_class,
                   coords_per_class_1se=selected.coords_mean_per_class,
                   val_F1_1se=f1_score(val42,chosen.predict(Zva),labels=labels_all,average='macro',zero_division=0),
                   val_F1_delta0=f1_score(val42,NSC().fit(Ztr,train42,0).predict(Zva),labels=labels_all,average='macro',zero_division=0))
        summary.append(row); print(json.dumps(row),flush=True)
    pd.DataFrame(runs).to_csv(HERE/'runs.csv',index=False)
    pd.DataFrame(summary).to_csv(HERE/'summary.csv',index=False)
    pd.DataFrame(curves).to_csv(HERE/'nsc_curves.csv',index=False)
    pd.DataFrame(assignment).to_csv(HERE/'assignments.csv',index=False)
    pd.DataFrame(sizes).to_csv(HERE/'cluster_composition.csv',index=False)
    # Existing partitions: descriptive reuse, not a new raw-vector run.
    old=pd.read_csv(BCV/'representation_analysis/data/cluster_assignments.csv'); recovered=[]
    for (rep,k), g in old.groupby(['representation','k']):
        lab=g.set_index('image_id').loc[ids[va],'cluster_id'].values
        recovered.append(dict(model=rep,k=k,val_organ_AMI=ami(organ[va],lab),val_scale_AMI=ami(scale[va],lab),val_species_AMI=ami(man.species.values[va],lab)))
    pd.DataFrame(recovered).to_csv(HERE/'existing_partitions.csv',index=False)
    (HERE/'provenance.json').write_text(json.dumps(dict(input_sha256=digest,k=KS,seeds=SEEDS,n_train=int(sum(tr)),n_val=int(sum(va)),test_used=False,
        python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,protocol=__doc__),indent=2))
    s=pd.DataFrame(summary)
    fig,axes=plt.subplots(2,2,figsize=(11,7.2))
    for col,label in [('val_organ_AMI','Organ'),('val_scale_AMI','Shot scale (provisional)'),('val_species_AMI','Species')]:
        axes[0,0].errorbar(s.k,s[col],yerr=s[col+'_sd'],marker='o',capsize=3,label=label)
    axes[0,0].set(ylabel='Validation adjusted mutual information',xlabel='K',title='What each partition aligns with');axes[0,0].legend(fontsize=8)
    axes[0,1].errorbar(s.k,s.val_silhouette,yerr=s.val_silhouette_sd,marker='o',capsize=3,label='Silhouette')
    axes[0,1].plot(s.k,s.seed_ARI_mean,marker='s',label='Mean pairwise seed ARI')
    axes[0,1].set(xlabel='K',ylabel='Score',title='Geometry and stability (validation)');axes[0,1].legend(fontsize=8)
    for k,g in pd.DataFrame(curves).groupby('k'):
        line,=axes[1,0].plot(g.coords_mean_per_class,g.cv_macroF1,label=f'K={k}')
        q=s[s.k==k].iloc[0];axes[1,0].scatter(q.coords_per_class_1se,q.cv_F1_1se,color=line.get_color(),s=20)
    axes[1,0].set(xlabel='Mean PCA coordinates per class (of 64)',ylabel='5-fold group-CV macro-F1 (absolute)',ylim=(0,1),title='NSC approximates fixed seed-42 partitions')
    axes[1,0].invert_xaxis();axes[1,0].legend(fontsize=8,ncol=2)
    axes[1,1].plot(s.k,s.coords_1se,marker='o',label='Any class')
    axes[1,1].plot(s.k,s.coords_per_class_1se,marker='s',label='Mean per class')
    axes[1,1].set(xlabel='K',ylabel='Retained PCA coordinates (of 64)',ylim=(0,66),title='1-SE sparsity depends on partition granularity');axes[1,1].legend(fontsize=8)
    fig.suptitle('EfficientNet K sensitivity: train-fitted clustering; no test data used',fontsize=12)
    fig.tight_layout();fig.savefig(HERE/'k_sensitivity.png',dpi=140);plt.close(fig)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
