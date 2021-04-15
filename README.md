# Détection des émotions en temps réel à l'aide du deep learning.

## Introduction

Ce projet vise à classer l'émotion sur le visage d'une personne dans l'une des "sept catégories", en utilisant des réseaux de neurones convolutifs profonds. Le modèle est formé sur l'ensemble de données. Cet ensemble de données se compose de 28716 images de visage en niveaux de gris et de taille 48x48 avec "sept émotions" - en colère, dégoûté, craintif, heureux, neutre, triste et surpris.

## Dépendances:

* Pour installer les packages requis, exécutez:
pip installer numpy
pip installer des pandas
pip installer tensorflow
pip installer keras
pip installer opencv-python

* Python 3, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)


## Utilisation de base:

* Pour cloner le projet sur votre ordinateur à partir d'un terminal:
```bash
git clone https://github.com/atulapra/Emotion-detection.git
cd mon_dossier
```

Le projet est compatible avec `tensorflow-2.0` et utilise l'API Keras en utilisant la bibliothèque` tensorflow.keras`.

*Pour télécharger le model pré-entrainé 'model.h5' suivez ce lien:
https://drive.google.com/file/d/1HafN_24_J81g_F1aEC_IifvRXDyl2vvp/view?usp=sharing


* Cette implémentation détecte par défaut les émotions sur tous les visages dans le flux de la webcam. Avec un CNN à 4 couches, la précision du test a atteint 63,2% en 50 époques.

![Accuracy plot](plot.png)

## Data Preparation (optional)

* The [original FER2013 dataset in Kaggle](https://www.kaggle.com/deadskull7/fer2013) is available as a single csv file. I had converted into a dataset of images in the PNG format for training/testing and provided this as the dataset in the previous section.

* In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. I have provided the code I wrote for data preprocessing in the `dataset_prepare.py` file which can be used for reference.

## Algorithm

* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.

## Example Output

![Mutiface](imgs/multiface.png)

## References

* "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
   Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,  
   X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
   M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
   Y. Bengio. arXiv 2013.
