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

Advisor: Daniel Fojo

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
6. [Results](#results)
    1. [Experiment 1: Single-artist training](#experiment_1)
    2. [Experiment 2: Specific genre training](#experiment_2)
    3. [Experiment 3: Conditional lyrics generation](#experiment_3)
    4. [Experiment 4: T5 model](#experiment_4)
7. [Conclusions](#conclusions)
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
* Define the main model architecture
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
    <img src="https://drive.google.com/uc?export=view&id=1jgmyVjKrc69KLUzmZw7j2BYIghZrDnZL" alt="Image" style="width: auto; max-width: 50%; height: 80px; display: inline-block;" title="Image" />
  </a>
  
  <a href="https://drive.google.com/uc?export=view&id=1N2ui7rYVl6WPUAgzuMgFe7TU2c_MGm56">
    <img src="https://drive.google.com/uc?export=view&id=1N2ui7rYVl6WPUAgzuMgFe7TU2c_MGm56" alt="Image" style="width: auto; max-width: 50%; height: 80px; display: inline-block;" title="Image" />
  </a>
  
  <a href="https://drive.google.com/uc?export=view&id=1LClGQxV6tDbLHU4dEowivvrehZPXnkHB">
    <img src="https://drive.google.com/uc?export=view&id=1LClGQxV6tDbLHU4dEowivvrehZPXnkHB" alt="Image" style="width: auto; max-width: 40%; height: 80px; display: inline-block;" title="Image" />
  </a>
  
  <a href="https://drive.google.com/uc?export=view&id=1gq6dYc2tmIJV2bvIZDrq2TaokTSYYm-j">
    <img src="https://drive.google.com/uc?export=view&id=1gq6dYc2tmIJV2bvIZDrq2TaokTSYYm-j" alt="Image" style="width: auto; max-width: 50%; height: 80px; display: inline-block;" title="Image" />
  </a>
</p>
<p align="right"><a href="#toc">To top</a></p>

## 4. General Architecture <a name="architecture"></a>
The development of advanced language models has brought significant changes to tasks like lyrics generation in natural language processing. These models, based on transformer architectures, have shown impressive skills in understanding and creating meaningful text that makes sense in different contexts. GPT, one of these models, has received a lot of attention because of its outstanding performance and flexibility. We have chosen to utilize GPT-2, which is the most recent version of the GPT models accessible on the Hugging Face platform.

GPT-2 consists of a stack of transformer layers, including both an encoder and a decoder. The encoder processes the input text and extracts its contextual representation, while the decoder generates new text based on the learned representations. This architecture allows GPT-2 to effectively capture the relationships between words and generate coherent and contextually relevant text.

<p align="left">
  <a href="https://drive.google.com/uc?export=view&id=1phbpF208uCZM_w4tmRwuD0BA9IdGb68G">
    <img src="https://drive.google.com/uc?export=view&id=1phbpF208uCZM_w4tmRwuD0BA9IdGb68G" alt="Image" style="width: 500px; height: 500px; display: inline-block;" title="Image" />
  </a>
</p>
<p align="right"><a href="#toc">To top</a></p>

## 5. Preprocessing the data set <a name="dataset_preprocess"></a>
Overall, the preprocessing steps involve:

* extracting the dataset
* removing non-English authors to ensure language consistency
* cleaning and formatting the lyrics data to eliminate unwanted artifacts
* tokenizing the datasets for further processing, setting a maximum context length 
<p align="right"><a href="#toc">To top</a></p>

## 6. Results <a name="results"></a>
    
### 6.1 Experiment 1: Single-artist training <a name="experiment_1"></a> 
We trained on about 100 lyrics by a single artist (exact amount depending on the number of lyrics  available in the dataset).
The main tendency that we observed is that the limitation in the size of the dataset led to overfitting. Experiments were conducted with different learning rate, all more or less leading to a similar result. 

The problems that we encountered in the generated lyrics were also mostly due to the small size of the dataset - predisposition to word repetition and to generating truncated lines or lines consisting of one word. We tried to address this issue in post processing by introducing a __post_process function that cleans up the generated sequences of lyrics by removing redundant line breaks, and removes consecutive duplicated words using the __remove_consecutive_duplicates helper function.

TODO: links to report/charts/screenshots of obtained resuts???? (weights and biases or other)
    

### 6.2 Experiment 2: Specific genre training <a name="experiment_2"></a>
Now we are training on larger amounts of data - a set of lyrics of a certain genre (determined by an argument specified in argparse) containing of several thousands of songs.
Results: we observed a decrease in overfitting issues, indicating a better generalization capability of the model. The generated lyrics showed reasonable quality and coherence, making more sense in the context of the chosen genre.
At this stage is became more difficult to complete training with the computational resources we had (without GPUs). 

TODO: links to report/charts/screenshots of obtained resuts???? (weights and biases or other)
<p align="right"><a href="#toc">To top</a></p>

### 6.3 Experiment 3: Conditional lyrics generation <a name="experiment_3"></a>
Training with a full dataset to generate song lyrics similar to those of a specific artist. The dataset (one of the two available) and the artist are determined by arguments specified in argparse.
Results: Lyrics of enhanced quality and coherence (though there is still quite a bit of room for improvement).
The main issue was again the lack of computational resources.

TODO: links to report/charts/screenshots of obtained resuts???? (weights and biases or other) 
<p align="right"><a href="#toc">To top</a></p>

### 6.4 Experiment 4: T5 model <a name="experiment_4"></a>
T5 (Text-To-Text Transfer Transformer) is a transformer-based language model developed by Google Research. Unlike GPT-2, which is primarily designed for autoregressive language generation, T5 follows a "text-to-text" framework. Instead of generating lyrics word by word like GPT-2, T5 is trained to map an input text prompt to an output text sequence.
T5 was trained on a large and diverse collection of publicly available text data from the internet. The specific details regarding the exact number of data, tokens, and parameters used for training T5 have not been disclosed publicly by Google Research. However, it is known to be a large-scale model with billions of parameters.

In our experiment due to the lack of time and computing power we chose to train the T5 model only on single-artist data.
The results observed were mainly similar to those we obtained with GPT-2 - overfitting and rather low quality of generated lyrics.

<a href="https://drive.google.com/uc?export=view&id=1VdnrzFDsd-yFs99O8ujCDiWPRzJFdoow" align="left">
  <img src="https://drive.google.com/uc?export=view&id=1VdnrzFDsd-yFs99O8ujCDiWPRzJFdoow" alt="Image" style="width: auto; max-width: 100%; height: 160px; display: inline-block;" title="Image" />
</a>


TODO: links to report/charts/screenshots of obtained resuts???? (weights and biases or other) 
<p align="right"><a href="#toc">To top</a></p>
