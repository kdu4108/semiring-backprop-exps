# Generalizing Backpropagation for Gradient-Based Interpretability
Hi there! This repo contains the experimentation systems and analysis for the paper ["Generalizing Backpropagation for Gradient-Based Interpretability"](https://arxiv.org/abs/2307.03056) by Du et al., published at ACL 2023.

### 🚧 Work-in-progress 🚧
Hi there! Please note that while this repo contains all preprocessing, experimentation, and analysis code to fully reproduce the results in the paper (as of July 2023), refactoring it to be *easily accessible* is still a work-in-progress. Thanks for your patience, and please open an issue on Github if you have any questions or concerns. I will soon be opening Github issues to track the specific tasks required to make this code more easily accessible and runnable.

Current priorities include:
*    Refactoring the experiment code and renaming scripts for consistency and clarity
*    Improved documentation describing the entry points for each experiment and any manual steps
*    Documenting which experiment code corresponds to which results in the paper
*    Dockerizing appropriate experiments

### Getting Started
1. Make sure you have an editable install of `brunoflow` (https://github.com/kdu4108/brunoflow). This library implements backpropagation such that it is easily extensible for other semirings.
2. Make sure you have an editable install of the brunoflow-modified `transformers` (https://github.com/kdu4108/transformers). This is necessary because transformers does not support brunoflow models and therefore I needed to reimplement brunoflow-based BERT models in this fork of the transformers library.
2. `conda install wandb`. We use Weights & Biases (https://wandb.ai/site) for experiment tracking. You will probably need to make an account in order to run these scripts (or else you'll need to comment out wandb-related code in the scripts/notebooks.)
3. `conda install papermill`. Since some of the experimentation code is written in Jupyter Notebooks, we use papermill (https://www.google.com/search?q=papermill&oq=papermill&aqs=chrome.0.69i59j0i512l2j0i10i512j0i512l2j0i10i30j0i30l3.831j0j7&sourceid=chrome&ie=UTF-8) as a programmatic executor for the notebooks. This enables us to run *parameterized* Jupyter Notebooks from top-to-bottom (like scripts) and saves them for provenance.
4. `pre-commit install` for some nice auto-linting/formatting upon committing.

### Datasets
For this paper, we constructed various synthetic datasets in addition to standard NLP datasets.
For the implementation of synthetic datasets, see the `preprocessing/datasets.py`.
For the subject-verb agreement (SVA) dataset, see https://github.com/yoavg/bert-syntax and our additional preprocessing/subsampling code in `preprocessing/lgd_sva/subsample_lgd_data.ipynb`.
Note that you will need to download the dataset from the repo linked above into the path `data/lgd_sva/lgd_dataset.tsv` for the subsampling script to work out of the box.

### Running experiments
**Validating top gradient path on a synthetic task (section 5.1)**
*    Task: `FirstTokenRepeatedOnce`

*    Experiment code entry point: `analyze_kvq_synthetic.py`

*    Analysis code: `analysis/synthetic_bert_max_grad.ipynb`

**Analyzing top gradient path of BERT on SVA (section 5.2)**
*    Task: subject-verb Agreement (SVA)

*    Experiment code entry point: `analyze_kvq.py`

*    Analysis code: `analysis/sva_max_grad.ipynb`

**Comparing gradient graph entropy vs task difficulty (section 5.3)**
*    Task: various synthetic tasks (see Appendix A of the paper).

*    Experiment code entry point: `compute_mdl.py`

*    Analysis code: `analysis/analyze_entropy_vs_mdl_synthetic.ipynb`

**Comparing gradient graph entropy vs example difficulty (appendix B.2)**
*    Task: subject-verb Agreement (SVA)

*    Experiment code entry point: `eval_bert.py`

*    Analysis code: `analysis/sva_entropy.ipynb`

**Sanity check for gradient graph entropy vs model complexity (appendix B.1)**
*    Task: `FFOOM`

*    Experiment code entry point: `python pm_train_mlp.py -p train_ffoom_mlp -hs 64 -s 6 -d FFOOM -k
'{"num_points": 10000}'`

*    Analysis code: `analysis/analyze_ffoom_entropy.ipynb`

### Limitations
*    Since brunoflow is not optimized for execution speed, it faces practical computational and memory constraints. We aim to recruit assistance to reimplement semiring-backpropagation in a more popular, GPU-accelerated library such as Pytorch or JAX to promote more practical use-cases of this method.
*    Again due to computational constraints from brunoflow's implementation, we run some experiments distributed over our slurm cluster. For those who are interested, we share those scripts in the `slurm/` directory.
