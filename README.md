With Conda

Create a conda environment by running

conda create --name project python=3.8
Then, activate the environment

conda activate project
and install the dependencies

pip install -r requirements.txt

# Lyrics Generator

Final project for the [UPC Postgraduate Course Artificial Intelligence with Deep Learning](https://www.talent.upc.edu/ing/estudis/formacio/curs/310400/postgraduate-course-artificial-intelligence-deep-learning/), edition Spring 2023

Team: Daniel Losada Molina, Pau Rosell Civit, Svetlana Kazakova

Advisor: Dani Fojo

GitHub repository: [https://github.com/DanielLosada/Transformers---Lyrics-Generator](https://github.com/DanielLosada/Transformers---Lyrics-Generator)

## Table of Contents <a name="toc"></a>
1. [Introduction](#intro)
    1. [Motivation](#motivation)
    2. [Project Goals](#goals)
    3. [Milestones](#milestones)
2. [Data Set](#dataset)
3. [Working Environment](#working_env)
4. [General Architecture](#architecture)
5. [Preprocessing the data set](#dataset_preprocess)
6. [Results and results improvement](#results)
    1. [Experiment 1: First train with the full dataset](#experiment_1)
    2. [Experiment 2: Overfitting with one chunk](#experiment_2)
    3. [Experiment 3: Long run in Google VM 2](#experiment_3)
    4. [Experiment 4: Multiple GPUs with WER](#experiment_4)
    5. [Experiment 5: Train with feature extractor](#experiment_5)
7. [Web Application](#web_app)
8. [Conclusions](#conclusions)
9. [Imagine one month more...](#1_month)
10. [Next Steps](#next_steps)
11. [References](#references)

## 1. Introduction <a name="intro"></a>
Lyrics generation, the task of automatically generating song lyrics using deep learning techniques, has gained significant attention in recent years. With the advancements in natural language processing and deep learning, generating creative and coherent lyrics has become an intriguing but still a challenging task. This project aims to explore and address these challenges by leveraging state-of-the-art deep learning models and fine-tuning them on suitable datasets. 

By analyzing the generated lyrics' quality, we can gain insights into the potential and limitations of deep learning models in the realm of lyrics generation.
<p align="right"><a href="#toc">To top</a></p>

### 1.1 Motivation <a name="motivation"></a>
Our motivation for this project is driven by two main factors: the desire to explore cutting-edge technologies and the fascination with the creative possibilities of lyrics generation. LLMs have shown impressive abilities in understanding and generating human-like text. By working on lyrics generation, we aim to dive deeper into these technologies and understand their potential for creative text generation.
<p align="right"><a href="#toc">To top</a></p>

### 1.2 Project Goals <a name="goals"></a>
* Attempt to generate lyrics with GPT-2 and T5 based models
* Analysis of the results
* Suggestions for further improvement
<p align="right"><a href="#toc">To top</a></p>

### 1.3 Milestones <a name="milestones"></a>
We have established the following key milestones:
* Do a general research on the subject
* Find suitable data sets
* Define the main model archirecture
* Preprocess the data and implement the model
* Train the model
* Analyse the obtained results and postprocess
* Try a different model architecture (optional)
* Make suggestions for further improvement
<p align="right"><a href="#toc">To top</a></p>

## 2. Data Set <a name="dataset"></a>
There are three primary methods for acquiring a dataset:

1. Extracting data from websites using tools like BeautifulSoup or Scrapy.
2. Utilizing the Genius API to gather data.
3. Accessing pre-existing datasets from platforms like Kaggle or other similar sources.

After trying all of them, we have opted for using datasets from Kaggle.
We have chosen the following 2 datasets:
* https://www.kaggle.com/datasets/mervedin/genius-lyrics
* https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres
<p align="right"><a href="#toc">To top</a></p>

## 3. Working Environment <a name="working_env"></a>
[PyTorch](https://pytorch.org/) is used as the main framework.
We started out in [Google Colab](https://colab.research.google.com/) as it was fast and easy for us to access. Then we passed on to training locally and via a VM instance on [Google Cloud](https://cloud.google.com/) but we had a problem with access to GPUs on Google Cloud therefore we couldn't complete our training there. To view the results we used [wandb](https://wandb.ai/site). 

<p align="middle">
  <a href="https://drive.google.com/uc?export=view&id=1jgmyVjKrc69KLUzmZw7j2BYIghZrDnZL">
    <img src="https://drive.google.com/uc?export=view&id=1jgmyVjKrc69KLUzmZw7j2BYIghZrDnZL" alt="Image" style="width: auto; max-width: 50%; height: 80px" title="Image" />
  </a>
</p>
<p align="right"><a href="#toc">To top</a></p>


