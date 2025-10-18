### Thank you for your interest. This repository contains the code for **On the Discrimination and Consistency for Exemplar-Free Class Incremental Learning** accepted at IJCAI-25.

#### Installation & Requirements

The current version of the codes has been tested with Python 3.8.19 on both Windows and Linux operating systems with the following requirements:

- 1x RTX 3080
- cuda==12.7
- numpy==1.24.3
- scipy==1.10.1
- scikit-learn==1.3.2
- torch==2.0.0
- torchvision==0.15.0
- tensorboardx==2.6.2.2
- diffdist==0.1

Please install the necessary packages.



#### Prepare data

For the CIFAR-100 experiment, you can simply run the code, and the dataset will be automatically downloaded to the **./data** folder. 

For the TinyImageNet dataset, you can download it from Link https://drive.google.com/file/d/1Q9zApR-zcOgMvG3YEsEWHIjfLPH3L7y3/view?usp=sharing and place it in the **./data** folder. 

For the ImageNet-Subset dataset, you can similarly download it from Link https://www.kaggle.com/datasets/arjunashok33/imagenet-subset-for-inc-learn and place it in the **./data** folder.



#### How to Test

You can first download the pre-trained weights from link https://drive.google.com/file/d/1m9TPJd8atgdQowZJr7YfgD9Zc7nkZkF1/view?usp=sharing and place the **logs** folder in your project directory. Then you can run:
`sh test_cifar100_10t.sh`
`sh test_timagenet_10t.sh`
`sh test_imagenet100_10t.sh`
to reproduce our results respectively.



#### How To Retrain

Please run:

`sh train_cifar100_10t.sh`
`sh train_timagenet_10t.sh`
`sh train_imagenet100_10t.sh`

to retrain three different dataset tasks from scratch.



#### Citation

If you find our work helpful, you may cite our articles:

```
@inproceedings{wang2025discrimination,
  title     = {On the Discrimination and Consistency for Exemplar-Free Class Incremental Learning},
  author    = {Wang, Tianqi and Guo, Jingcai and Li, Depeng and Chen, Zhi},
  booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence},
  pages     = {6424--6432},
  year      = {2025}
}
```



#### Acknowledgments

Our code draws upon the prior work of 

[CLOM]: https://arxiv.org/abs/2203.09450

. We express our gratitude for their work.

