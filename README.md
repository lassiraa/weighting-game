# The Weighting Game: Evaluating Quality of Explainability Methods

Standalone code for the paper: [The Weighting Game: Evaluating Quality of Explainability Methods](https://arxiv.org/abs/2208.06175) by Lassi Raatikainen and Esa Rahtu.


## Abstract

The objective of this paper is to assess the quality of explanation heatmaps for image classification tasks. To assess the quality of explainability methods, we approach the task through the lens of accuracy and stability.

In this work, we make the following contributions. Firstly, we introduce the Weighting Game, which measures how much of a class-guided explanation is contained within the correct class' segmentation mask. Secondly, we introduce a metric for explanation stability, using zooming/panning transformations to measure differences between saliency maps with similar contents.

Quantitative experiments are produced, using these new metrics, to evaluate the quality of explanations provided by commonly used CAM methods. The quality of explanations is also contrasted between different model architectures, with findings highlighting the need to consider model architecture when choosing an explainability method.


## Software implementation

> TODO

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/lassiraa/weighting-game.git


## Dependencies

You'll need a working Python (>=3.9) environment to run the code. The dependencies can be installed with pip using:

    pip install -r requirements.txt

Or with conda using:

    conda install --file requirements.txt

## Cite this work
If you use this code for your project please consider citing us:
```
TODO
```