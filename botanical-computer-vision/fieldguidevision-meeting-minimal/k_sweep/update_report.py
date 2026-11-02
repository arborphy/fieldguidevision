"""Refresh the absolute-F1 plot and append the K sensitivity section in report_en_standalone.html."""
from pathlib import Path
from io import BytesIO
import base64, re, zipfile
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REPORT=HERE.parent/'report_en_standalone.html'
def inline_png(data):return 'data:image/png;base64,'+base64.b64encode(data).decode()
def table(df):return '<div class=tw>'+df.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')+'</div>'
def main():
    s=REPORT.read_text()
    fig,ax=plt.subplots(figsize=(6.5,3.6))
    with zipfile.ZipFile(HERE.parent/'changed_files.zip') as z:
        for name,sub in [('EfficientNet-B0',''),('BioCLIP 2.5','bioclip/'),('DINOv3','dinov3/')]:
            c=pd.read_csv(BytesIO(z.read('botanical-computer-vision/meeting_minimal/results/'+sub+'exp1_nsc_curves.csv')))
            c=c[c.task.str.startswith('organ')]
            assert len(c)==33 and c.cv_macroF1.between(0,1).all()
            ax.plot(c.coords_mean_per_class,c.cv_macroF1,marker='o',ms=3,label=name)
    ax.set(xlabel='mean PCA coordinates kept per organ class (of 64)',ylabel='5-fold group-CV macro-F1 (train)',ylim=(0,1))
    ax.set_title('NSC organ task: accuracy vs sparsity, 3 models',fontsize=9)
    ax.legend(fontsize=8);ax.invert_xaxis();fig.tight_layout()
    buf=BytesIO();fig.savefig(buf,format='png',dpi=120);plt.close(fig)
    plot=f"<img src='{inline_png(buf.getvalue())}' alt='NSC 3 models: absolute 5-fold group-CV macro-F1 (0–1)'>"
    s,n=re.subn(r"<img src='data:image/png;base64,[^']+' alt='NSC 3 models[^']*'>",lambda m:plot,s)
    assert n==1
    a=pd.read_csv(HERE/'summary.csv')
    q=pd.DataFrame({'K':a.k,'organ AMI':a.val_organ_AMI,'scale AMI':a.val_scale_AMI,'species AMI':a.val_species_AMI,
                    'silhouette':a.val_silhouette,'seed ARI':a.seed_ARI_mean,'train clusters <10 (mean)':a.train_clusters_lt10})
    nsc=a[['k','delta_1se','coords_1se','coords_per_class_1se','cv_F1_1se','val_F1_delta0','val_F1_1se']].copy()
    nsc.columns=['K','1-SE delta','coords / 64','mean coords / class','CV F1 at 1-SE','val F1 delta=0','val F1 at 1-SE']
    old=pd.read_csv(HERE/'existing_partitions.csv')
    section='''<section id=e9><h2>9. K sensitivity: what does the number of clusters mean?</h2>
<div><span class='b v'>New EfficientNet runs</span> <span class='b r'>Existing BioCLIP / DINOv3 partitions</span></div>
<p><b>K is the granularity of a partition, not a measured number of biological categories.</b> Small K can merge distinct organs or views; larger K can split an organ by scale, species, morphology or individual plant. The 7 organ labels and 63 species do not imply K should equal 7 or 63. K-means groups embedding similarity, which may mix these factors.</p>
<h4>New experiment: K = 5, 10, 15, 20, 30, 50</h4>
<p>EfficientNet raw L2-normalized 1,280-D embeddings; Euclidean K-means fit only on 1,181 train images. Five seeds (42–46), ten initializations per seed. Frozen centroids assign 249 validation images from disjoint individual plants. No test images are used. This K=20 is a <b>new train-fitted partition</b>, so its numbers are not directly interchangeable with the original all-corpus K=20 results above. AMI adjusts label agreement for chance, avoiding the simple increase in raw purity or unadjusted mutual information that can accompany more clusters. Organ, provisional scale and species labels are three different interpretations of a partition.</p>
<p>Table values are means across five seeds; plotted error bars are seed SD, not sampling confidence intervals. Seed ARI averages the ten pairwise comparisons on the same validation images; it measures sensitivity to optimization, not population uncertainty.</p>'''+table(q)+f"<img src='{inline_png((HERE/'k_sensitivity.png').read_bytes())}' alt='K sensitivity: label alignment, seed stability, absolute group-CV F1, and sparsity'>"+'''
<h4>NSC grouping approximation at every K (all F1 values absolute)</h4>
<p>For each K, NSC approximates the fixed seed-42 K-means partition. PCA-64 is fit separately inside each individual-group CV fold; the final basis is fit on all train. A 33-point shrinkage grid and the 1-SE rule choose delta using five training-group folds. Validation is evaluated at delta=0 and the selected delta. Macro-F1 includes all K classes, with zero for undefined class scores. At K=50, only 47 of 50 clusters are represented in validation on average across seeds.</p>'''+table(nsc)+'''
<p><b>Interpretation:</b> K=5 has the strongest organ alignment among the tested values (AMI 0.520) and the highest silhouette (0.077); K=10 has the highest mean seed ARI (0.878). At K=50, organ AMI is 0.375, silhouette 0.038 and seed ARI 0.456; on average 4.2 training clusters have fewer than 10 images. More clusters lower training inertia, but do not automatically improve biological interpretation or reproducibility.</p>
<p>The original “K=20 grouping needs almost every PC” conclusion is <b>conditional on K and protocol</b>: here the 1-SE model keeps 48/64 coordinates at K=5, 57 at K=10, 63 at K=15, and all 64 at K=20/30/50. Even K=5 does not collapse to just a few PCs. F1 falls as K grows partly because the target becomes finer and some classes have little support; these are different target tasks, not a cross-K model leaderboard. The targets are derived from the same embeddings, so high grouping F1 demonstrates partition reconstruction, not independent biological prediction. The fixed teacher partition uses all training images; CV uncertainty is conditional on that partition.</p>
<h4>Existing BioCLIP / DINOv3 K=10/20/50 partitions</h4>
<p>These label-alignment numbers reuse checked-in earlier all-corpus partitions, restricted to the same validation image IDs for descriptive comparison. They are <b>not new train-fitted raw-vector experiments</b>. The available NPZ projection files contain only 2-D PCA/UMAP coordinates; the recovered raw vectors used by the older report are absent from this checkout. No new BioCLIP/DINOv3 geometry, seed-stability or NSC results are claimed here.</p>'''+table(old)+'''
<p>BioCLIP species AMI increases from 0.351 (K=10) to 0.630 (K=50), while organ AMI remains much lower (0.131–0.214). DINOv3 organ AMI peaks at K=20 among its three saved partitions (0.511), while species AMI increases to 0.357 at K=50. This is consistent with different embeddings splitting different factors at the same K; it does not establish an optimal K for either model.</p>
<p><b>Practical reading:</b> use K=5 and K=10 as coarse, more stable EfficientNet views; compare K=15/20 as intermediate resolution and K=30/50 as fine resolution with cluster-support checks. Keep several K values visible instead of treating 20 as a biological constant. Inspect representative and boundary images before naming a cluster. No single K is selected from test performance.</p>
<p class=note>Reproduce: <code>python botanical-computer-vision/fieldguidevision-meeting-minimal/k_sweep/run_k_sweep.py</code>, then <code>python botanical-computer-vision/fieldguidevision-meeting-minimal/k_sweep/update_report.py</code>. Scripts, per-seed metrics, assignments, cluster composition, full CV curves and provenance are saved in <code>k_sweep/</code>. Metric definitions: <a href='https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation'>scikit-learn clustering evaluation</a>.</p></section>'''
    if '<section id=e9>' in s:
        s,n=re.subn(r'<section id=e9>.*?</section>',lambda m:section,s,flags=re.S);assert n==1
    else:
        assert s.count('<section><h2>Reproduce</h2>')==1
        s=s.replace('<section><h2>Reproduce</h2>',section+'<section><h2>Reproduce</h2>')
        s=s.replace('<a href=#e8>appendix</a></nav>','<a href=#e8>appendix</a><a href=#e9>K sensitivity</a></nav>')
    REPORT.write_text(s)
    print('Updated absolute F1 and K sensitivity section.')
if __name__=='__main__':main()
