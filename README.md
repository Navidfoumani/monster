<div align="center"><img src="figs/Logo.png" style="width:400px"></div>

# MONSTER

***MONSTER: Monash Scalable Time Series Evaluation Repository***

[arXiv:2502.15122](https://arxiv.org/abs/2502.15122) (preprint)  
The datasets are hosted on **Hugging Face** and can be accessed here:
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Datasets-orange?style=for-the-badge&logo=huggingface)](https://huggingface.co/monster-monash)

<!-- [HuggingFace](https://huggingface.co/monster-monash) (data) -->

> <div align="justify">We introduce MONSTER&mdash;the <b>MON</b>ash <b>S</b>calable <b>T</b>ime Series <b>E</b>valuation <b>R</b>epository&mdash;a collection of large datasets for time series classification. The field of time series classification has benefitted from common benchmarks set by the UCR and UEA time series classification repositories. However, the datasets in these benchmarks are small, with median sizes of 217 and 255 examples, respectively. In consequence they favour a narrow subspace of models that are optimised to achieve low classification error on a wide variety of smaller datasets, that is, models that minimise variance, and give little weight to computational issues such as scalability. Our hope is to diversify the field by introducing benchmarks using larger datasets. We~believe that there is enormous potential for new progress in the field by engaging with the theoretical and practical challenges of learning effectively from larger quantities of data.</div>

Please cite as:
```bibtex
@article{dempster_etal_2025,
  author  = {Dempster, Angus and Foumani, Navid Mohammadi and Tan, Chang Wei and Miller, Lynn and Mishra, Amish and Salehi, Mahsa and Pelletier, Charlotte and Schmidt, Daniel F and Webb, Geoffrey I},
  title   = {MONSTER: Monash Scalable Time Series Evaluation Repository},
  year    = {2025},
  journal = {arXiv:2502.15122},
}
```

## Downloading Data

### <tt>hf_hub_download</tt>

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id = f"monster-monash/Pedestrian", filename = f"Pedestrian_X.npy", repo_type = "dataset")

X = np.load(path, mmap_mode = "r")
```

### <tt>load_data</tt>

```python
from datasets import load_dataset

dataset = load_dataset("monster-monash/Pedestrian", "fold_0", trust_remote_code = True)
```

(More to come...)

<div align="center">🦖</div>


