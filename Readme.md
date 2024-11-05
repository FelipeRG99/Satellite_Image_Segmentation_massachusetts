Deep learning based in U-net architecture for the Massachusetts dataset consisting in segmentation of buildings and roads

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
- Model(Build dataset) + Model(Road dataset)=='sep'. Two Binary Classification U-net
- Model(Build dataset + Road dataset)=='uni'. Multiclass U-net

Results
-----------------------
<p float="left">
  <img src="Img results/sep_23728930_15_result.png" width="45%" />
  <img src="Img results/uni_23728930_15_result.png" width="45%" />
</p>

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