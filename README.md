# BiT-Tobacco-800


[<img src="https://deepnote.com/buttons/launch-in-deepnote.svg">](https://deepnote.com/project/9045f0e2-4d02-4e64-b313-d70bdf35dcc5#%2FBiT-Tobacco-800%2Fbig_transfer_model.ipynb)

## Overview

This project explores Document Embedding Vectors for Image Clustering and Reverse Image Search for Large scale document EDA.
It is applicable to Entreprise type documents which are largely text based, black and white multi-page documents.
<br>

This project extends the [BiT Colab Notebook](https://colab.research.google.com/github/google-research/big_transfer/blob/master/colabs/big_transfer_pytorch.ipynb)
to demonstrate how embeddings can be extracted from the model to be used for unsupervised learning. The note explores the Big Transfer model built by Google as explained in [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370).
<br>
The data is preloaded in the `data` folder to make the notebook easy to run without needing to download any additional packages.
When initialised, the notebook will automatically run the set-up script in the background.

*Note: Deepnote currently runs only on Cloud CPUs, GPU is not yet available*


# Paper Analysis

##Abstract


*Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training deep neural networks for vision. We revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model on a target task. We scale up pre-training, and propose a simple recipe that we call Big Transfer (BiT). By combining a few carefully selected components, and transferring using a simple heuristic, we achieve strong performance on over 20 datasets. BiT performs well across a surprisingly wide range of data regimes — from 1 example per class to 1 M total examples. BiT achieves 87.5% top-1 accuracy on ILSVRC-2012, 99.4% on CIFAR-10, and 76.3% on the 19 task Visual Task Adaptation Benchmark (VTAB). On small datasets, BiT attains 76.8% on ILSVRC-2012 with 10 examples per class, and 97.0% on CIFAR-10 with 10 examples per class. We conduct detailed analysis of the main components that lead to high transfer performance.*
<br>

The goal of Big Transfer is to be the standard model to start any Computer Vision task, similar to the position of BERT in Natural Language Processing.
To achieve this, Google Brain collate very large datasets to pre-train the model from, with the models working very well out of the box on 1-Shot examples and easy to fine tune on other image types.

BiT has 3 model variants: <br>
- BiT-S models pre-trained on 1.28M images from ILSVRC-2012
- BiT-M models pre-trained on 14.2M images from ImageNet-21k
- BiT-L models pre-trained on 300M images from JFT-300M.

One interesting point mentioned in the paper is that increases to model size will only have an affect if the corresponding dataset is big enough. 
Conversely, there is little benefit in pre-training smaller models on larger datasets.

![Model Size Comparison](/figures/model_size.png "Model Size and Performance Comparison")