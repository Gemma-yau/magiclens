# MagicLens
Basic Information of selected paper
Paper Name: MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions\n
Authors: Kai Zhang, Yi Luan, Hexiang Hu, Kenton Lee, Siyuan Qiao, Wenhu Chen, Yu Su, Ming-Wei Chang\n
Venue: the International Conference on Machine Learning (ICML) 2024



## Setup
```
conda create --name magic_lens python=3.10
conda activate magic_lens
git clone https://github.com/google-research/scenic.git
cd scenic
pip install .
pip install -r scenic/projects/baselines/clip/requirements.txt
# you may need to install corresponding GPU version of jax following https://jax.readthedocs.io/en/latest/installation.html
# e.g.,
# # CUDA 12 installation
# Note: wheels only available on linux.
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # CUDA 11 installation
# Note: wheels only available on linux.
# pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Model Download
Download model via:
```
cd .. # in main folder `magiclens`
# you may need to use `gcloud auth login` for access, any gmail account should work.
gsutil cp -R gs://gresearch/magiclens/models ./
```

OR via [google drive](https://drive.google.com/drive/folders/1MXszMqIIh-yV7cYxWUxP7uHs9gfuTT3u)

### Data Preparation
Please follow each dataset folder in `./data`.

## Inference
```
python inference.py \
--model_size large \
--model_path ./models/magic_lens_clip_large.pkl \
--dataset circo

```

either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
