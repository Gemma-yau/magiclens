# MagicLens

This repo contains implementation of MagicLens. The code here uses Jax and Flax.
Note that the current implementation does not yet support training.
Refer to the [website](https://open-vision-language.github.io/MagicLens/) for dataset examples.

## Abstract

We introduce MagicLens, a series of self-supervised image retrieval models that support
open-ended instructions. The core thesis of MagicLens is that text
instructions can enable retrieving images with
richer relations beyond visual similarity. MagicLens is built on a
key novel insight: image pairs that naturally occur
on the same web pages contain a wide range of implicit relations (e.g., inside view of), and we
can bring those implicit relations explicit by synthesizing instructions via large multimodal models (LMMs) and large language models (LLMs).
Trained on 36.7M (query image, instruction, target image) triplets with rich semantic relations
mined from the web, MagicLens achieves comparable or better results on eight benchmarks of
various image retrieval tasks than prior state-of-the-art (SOTA) methods. Remarkably, it outperforms previous SOTA but with a 50× smaller
model size on multiple benchmarks. Additional
human analyses on a 1.4M-image unseen corpus
further demonstrate the diversity of search intents
supported by MagicLens.
![Intro image](https://open-vision-language.github.io/MagicLens/static/images/magiclens_overview.png)

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
