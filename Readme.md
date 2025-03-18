Deep learning based in U-net architecture for the Massachusetts dataset consisting in segmentation of buildings and roads. Comparison between Custom Unet and Unet from segmented models. 

Dataset
-----------------------
The used dataset is derived from Volodymyr Mnih's original Massachusetts Buildings/Roads Dataset and can be found in Kaggle for [Buildings](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset) and [Roads](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset). The dataset contains images 1500×1500 pixels in size, 1171 aerial images for the road dataset and 151 aerial image for the building dataset. Some of them use the same base image but not all, the road dataset images are taken from the state of Massachusetts and the building dataset are from the Boston area.


Info
-----------------------
The majority of the proyect has comments in spannish (my native language) cause is a for fun proyect and for learnning purpose.

[PreProcessing .tiff](Notebooks\Procesado_tiff_png.ipynb)
-----------------------
Instead of working with .tiff is more efective and is least expensive to store if i transform it to .png. In this process i also filter the images with artifacts (white cuts in the image). Then, i worked with the .png. This .png should be in a directory called Img and separated between building and road images. I cant´n upload it because is 5gb of images.

Trained Models
-----------------------
The trained models weigh too much (>300Mb) so they cannot be uploaded. To create these model weights, it is necessary to run the [Notebooks](Notebooks) and obtain these weights (Of course create a directory named Trained_Models).

[Main](Main.py)
-----------------------
There is two models, one of them consisting in the sum of two submodels:
- Model(Build dataset) + Model(Road dataset)=='sep'. Two Binary Classification U-net with segmentation_models
- Model(Build dataset + Road dataset)=='uni'. Multiclass U-net with segmentation_models
- Custom_Model(Build dataset) + Custom_Model(Road dataset)=='cus'. Two Binary Classification custom U-net

Results
-----------------------
Here you can see the results test for the U-net with segmentation_models and the custom U-net. Then, other results can be seen of the reconstruction of the prediction of the images using the mixed models explained before; Model(Build dataset) + Model(Road dataset), Model(Build dataset + Road dataset), Custom_Model(Build dataset) + Custom_Model(Road dataset), and that can be used in [Main](Main.py).

<div align="center">

| Model   | F1 (Test) | Accuracy (Test) |
|:-------:|:----------:|:---------------:|
| Unet_custom (Roads)    | 0.8423      | 0.9818         |
| Unet (Roads) | 0.853    | 0.9822          |
| Unet_custom (Build) | 0.8872     | 0.9319          |
| Unet (Build) | 0.88     | 0.9268          |

<div align="center">
  <p><strong>Unet sep</strong></p>
  <img src="Img results/sep_23728930_15.png" width="45%">
</div>

<div align="center">
  <p><strong>Unet Custom</strong></p>
  <img src="Img results/cus_23728930_15.png" width="45%">
</div>

<div align="center">
  <p><strong>Unet uni</strong></p>
  <img src="Img results/uni_23728930_15.png" width="45%">
</div>

<p float="left">
  <figure style="display:inline-block; margin-right: 10px;">
    <img src="Img results/sep_23728930_15_result.png" width="45%" />
    <figcaption>Unet sep</figcaption>
  </figure>
  <figure style="display:inline-block;">
    <img src="Img results/uni_23728930_15_result.png" width="45%" />
    <figcaption>Unet uni</figcaption>
  </figure>
</p>

<p float="left">
  <figure style="display:inline-block;">
    <img src="Img results/cus_23728930_15_result.png" width="45%" />
    <figcaption>Unet Custom</figcaption>
  </figure>
</p>

</div>

Acknowledgements
-----------------------
This dataset is derived from Volodymyr Mnih's original Massachusetts Buildings/Roads Dataset. Massachusetts Roads Dataset & Massachusetts Buildings dataset were introduced in Chapter 6 of his PhD thesis. If you use this dataset for research purposes you should use the following citation in any resulting publications:
``` 
@phdthesis{MnihThesis,
author = {Volodymyr Mnih},
title = {Machine Learning for Aerial Image Labeling},
school = {University of Toronto},
year = {2013}
}
``` 