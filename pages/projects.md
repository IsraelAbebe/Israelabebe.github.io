---
layout: default
title: Projects
subtitle: 
---

<p align='center'>
<img src="http://ghchart.rshah.org/IsraelAbebe" alt="2016rshah's Github chart" />
<br>
</p>

<hr>


<h2 align='center'> Projects I Worked On </h2>
--------------------------------------------------

1. ### MasakhaNEWS: News Topic Classification for African languages
> African languages are severely under-represented in NLP research due to lack of datasets covering several NLP tasks. While there are individual language specific datasets that are being expanded to different tasks, only a handful of NLP tasks (e.g. named entity recognition and machine translation) have standardized benchmark datasets covering several geographical and typologically-diverse African languages. In this paper, we develop MasakhaNEWS -- a new benchmark dataset for news topic classification covering 16 languages widely spoken in Africa. We provide an evaluation of baseline models by training classical machine learning models and fine-tuning several language models. Furthermore, we explore several alternatives to full fine-tuning of language models that are better suited for zero-shot and few-shot learning such as cross-lingual parameter-efficient fine-tuning (like MAD-X), pattern exploiting training (PET), prompting language models (like ChatGPT), and prompt-free sentence transformer fine-tuning (SetFit and Cohere Embedding API). Our evaluation in zero-shot setting shows the potential of prompting ChatGPT for news topic classification in low-resource African languages, achieving an average performance of 70 F1 points without leveraging additional supervision like MAD-X. In few-shot setting, we show that with as little as 10 examples per label, we achieved more than 90% (i.e. 86.0 F1 points) of the performance of full supervised training (92.6 F1 points) leveraging the PET approach.
>>[paper](https://arxiv.org/pdf/2304.09972v1.pdf) | [code](https://github.com/masakhane-io/masakhane-news)






2. ### Natural Language Processing in Ethiopian Languages: Current State, Challenges, and Opportunities
> This survey delves into the current state of
natural language processing (NLP) for four
Ethiopian languages: Amharic, Afaan Oromo,
Tigrinya, and Wolaytta. Through this paper,
we identify key challenges and opportunities
for NLP research in Ethiopia. Furthermore,
we provide a centralized repository on GitHub
that contains publicly available resources for
various NLP tasks in these languages. This
repository can be updated periodically with
contributions from other researchers. Our objective is to identify research gaps and disseminate the information to NLP researchers interested in Ethiopian languages and encourage future research in this domain.
>>[PDF here](https://arxiv.org/pdf/2303.14406.pdf)



2. ### Masakhane-Afrisenti at SemEval-2023 Task 12: Sentiment Analysis using Afro-centric Language Models and Adapters for Low-resource African Languages
> In this paper, we describe our submission
for the AfriSenti-SemEval Shared Task 12 of
SemEval-2023. The task aims to perform
monolingual sentiment classification (sub-task
A) for 12 African languages, multilingual sentiment classification (sub-task B), and zeroshot sentiment classification (task C). For subtask A, we conducted experiments using classical machine learning classifiers, Afro-centric
language models, and language-specific models. For task B, we fine-tuned multilingual pretrained language models that support many of
the languages in the task. For task C, we
used we make use of a parameter-efficient
Adapter approach that leverages monolingual
texts in the target language for effective zeroshot transfer. Our findings suggest that using
pre-trained Afro-centric language models improves performance for low-resource African
languages. We also ran experiments using
adapters for zero-shot tasks, and the results
suggest that we can obtain promising results
by using
>>[PDF here](https://arxiv.org/pdf/2304.06459.pdf)

3. ### Exploring Data Imbalance and Modality Bias inHateful Memes
> Multi-modal memes which consist of an image and text are very popular on social
media but can sometimes be intentionally or unintentionally hateful. Understanding
them and if they are hateful frequently requires to consider image and text jointly.
Naturally, hateful memes appear less frequent than non-hateful ones, creating a
data imbalance in addition to modality biases present between the language and
visual modality. In this work, we study the Hateful Memes dataset and evaluate
several approaches to reduce data imbalance. In our experiments we show that
simple dataset balancing and image augmentation can reduce the most concerning
error, namely overlooking hateful content, significantly (175 to 112 errors), at a
slight increase of overall accuracy.<br>
>>[PDF here](https://drive.google.com/file/d/1LarvIKEkYNu9lZPt97diU_OmGieg8Sow/view)

4.  ### An Amharic News Text classification Dataset
 >  In NLP, text classification is one of the primary problems we try to solve and its uses in language analyses are indisputable. The lack of labeled training data made it harder to do these tasks in low resource languages like Amharic. The task of collecting, labeling, annotating, and making valuable this kind of data will encourage junior researchers, schools, and machine learning practitioners to implement existing classification models in their language. In this short paper, we aim to introduce the Amharic text classification dataset that consists of more than 50k news articles that were categorized into 6 classes. This dataset is made available with easy baseline performances to encourage studies and better performance experiments.<br>
    >>[arxiv.org/abs/2103.05639](https://arxiv.org/abs/2103.05639)

5. ### MasakhaNER: Named Entity Recognition for African Languages
    > We take a step towards addressing the under-representation of the African continent in NLP research by creating the first large publicly available high-quality dataset for named entity recognition (NER) in ten African languages, bringing together a variety of stakeholders. We detail characteristics of the languages to help researchers understand the challenges that these languages pose for NER. We analyze our datasets and conduct an extensive empirical evaluation of state-of-the-art methods across both supervised and transfer learning settings. We release the data, code, and models in order to inspire future research on African NLP.<br>
>>[https://arxiv.org/abs/2103.11811](https://arxiv.org/abs/2103.11811)

6. ### Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets

    > With the success of large-scale pre-training and multilingual modeling in Natural Language Processing (NLP), recent years have seen a proliferation of large, web-mined text datasets covering hundreds of languages. We manually audit the quality of 205 language-specific corpora released with five major public datasets (CCAligned, ParaCrawl, WikiMatrix, OSCAR, mC4). Lower-resource corpora have systematic issues: At least 15 corpora have no usable text, and a significant fraction contains less than 50% sentences of acceptable quality. In addition, many are mislabeled or use nonstandard/ambiguous language codes. We demonstrate that these issues are easy to detect even for non-proficient speakers, and supplement the human audit with automatic analyses. Finally, we recommend techniques to evaluate and improve multilingual corpora and discuss potential risks that come with low-quality data releases.<br>
>>[https://arxiv.org/abs/2103.12028](https://arxiv.org/abs/2103.12028)

7. ### Plant Disease Detection using Deep Learning
    > . Train and Evaluate different DNN Models for plant disease detection problem <br>
    > . To tackle the problem of scarce real-life representative data, experiment with different generative networks and generate more plant leaf image data <br>
    > . Implement segmentation pipeline to avoid misclassification due to unwanted input <br>
>>[Github repository](https://github.com/IsraelAbebe/plant_disease_experiments)

8. ### Image Retrieval in Pytorch
    > This Project implements image retrieval from large image dataset using different image similarity measures based on the following two approaches. <br>
        >>  Based on Siamese Network <br>
        >>  Using Resnet pre-trained Network to extract features and store them based on LSH simmilarity to get faster responce for large dataset. <br>
>>[Github repository](https://github.com/IsraelAbebe/Image-retrieval-in-pytorch)

9. ### Amharic Online Handwriting Recognition
    > Using the touch screen capabilities of handhelds input method which is similar to handwriting is
            the best way to replace the current text entry approach. most people learn writing by using a pen
            and a pencil not a keyboard so writing on a touchscreen device like you write on a piece of paper
            is the natural and easiest way for the users .users can stroke their hand on a given canvas and the
            application reads what they write .
            Online handwriting recognition (OHWR) is getting renewed interest as it provides data entry
            mechanism that is similar to natural way of writing.
            This project mainly focuses on using this technique to solve the problem mentioned above. The
            online handwriting recognition project is not new idea .in recent years many researchers are
            adopting this method for their language use. 
        <br>
>>[Project pdf](https://drive.google.com/file/d/1Ez0lWNhFe_WTk24bC-CBML7lsYPnKI3T/view?usp=sharing)


10. ### Cassava Disease Classification
    > This project aims to detect cassava diseases in a dataset of 5 fine-grained cassava leaf disease categories with 9,436 annotated images and 12,595 unlabeled images.<br>

    >>[Github repository](https://github.com/IsraelAbebe/cassava_disease_classification)

11. ### Amharic-NLP Projects
    > Some NLP Projects for my Native language [Amharic](https://en.wikipedia.org/wiki/Amharic) <br>
    >>[GitHub repository](https://github.com/IsraelAbebe/Amharic-NLP)