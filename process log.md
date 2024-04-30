># April
>>- **index** : 1
>>- **date** : 16/04/2024
>>- **summary** : \
    &nbsp;*1.* read paper which is regarding dataset "Fakeddit" \
    &nbsp;*2.* download the dataset \
    &nbsp;*3.* create git repository \
    &nbsp;*4.* learn Deep NLP lab1 exe1 
>>- **note** : \
    &nbsp;*1.Fakeddit:*  1,063,106 samples in total, 60% multimodal samples(have both text-image) \
    &nbsp;*2.source:* reddit(22 subreddits) \
    &nbsp;*3.labels:* 2-label(T/F) / 3-label(T/F/fake & contain true text) / 6-label( T/ F&Satire,Parody / F&Misleading Context / Manipulated Content / F&False Connection / F&Imposter Content) \
    &nbsp;*4.lab:* use models: FastText, LangID, langdetext to predict label of languages 
>---
>>- **index** : 2
>>- **date** : 17/04/2024
>>- **summary** : \
    &nbsp;*1.* read paper "Fact-Checking Meets Fauxtography: Verifying Claims About Images" \
    &nbsp;*2.* download the dataset Fauxtography ---- **can't unzip** \
>>- **note** : \
    &nbsp;*1.Fauxtography:* comes from two sources: **The snopes**, **The Reuters** \
    &nbsp;*2.Features:* 13 from "Google tags","URL domains", "URL categories", "True/False/Mixed media percentage", "Known media percentage", "True/False/Mixed media titles"\
    &nbsp;*3.lab:* use Natural Language Toolkit(nltk), spacy to count word token 
>>
>>| name | data type | label | size|
>>|---|---|---|---|
>>|The Snopes Dataset  | image-text pair | True/Fake | 197 True + 641 Fake|
>>|The Reuters Dataset | image-text pair | True only | 395 True|
>---
>>- **index** : 3
>>- **date** : 18/04/2024
>>- **summary** : \
    &nbsp;*1.* read paper "Text Image-CNN" 
>>- **note** : \
    &nbsp;*1.* Deception detection: Scientific fraud, fake news, false tweets...
    &nbsp;*2.* Sove deception detection approaches: 1.linguistic approach(NLP techniques) 2. network appriach(analyze the network structure and behaviors) 3.Neural Network
>---
>>- **index** : 4
>>- **date** : 19/04/2024
>>- **summary** : \
    &nbsp;*1.* read paper "Text Image-CNN" 
    &nbsp;*2.* download the dataset
>>- **note** : \
    &nbsp;*1.Dataset:* 20,015 samples= 11,941 fake+8074 true;\
    &nbsp;*2.source:* **Fake news**: more than 240 websites on Kaggle; **True news**: well known news websites\
    &nbsp;*3.Features:* Text/Image -> explicit features + latent features\
    &nbsp;*4.:*
>>
>>|type|explicit features|latent features|
>>|---|---|---|
>>|Text|derived from the statistic of the news text|extract the word-level embedding by CNN |
>>|Image|extract the resolution, number of faces...| CNN|
>---
>>- **index** : 5
>>- **date** : 20/04/2024
>>- **summary** : \
    &nbsp;*1.* read paper "Detection and visualization of misleading content on Twitter" 
    &nbsp;*2.* download the dataset
>>- **note** : \
    &nbsp;*1.:* Two-step classification model based on a novel semi-supervised learning scheme\
    &nbsp;*2.:* the result of first 2 classifier will be used to tune the second layer\
    &nbsp;*3.Features:* extract tweet-based and user-based features, one of them will be used in one of 2 classifiers in first layer, then combine these two results\
>---
>>- **index** : 6
>>- **date** : 22/04/2024
>>- **summary** : \
    &nbsp;*1.* read paper "Detection and visualization of misleading content on Twitter" 
>>- **note** : \
>---
> ## Week 2
>>- **index** : 7
>>- **date** : 23/04/2024
>>- **summary** : \
    &nbsp;*1.* read paper "Bert" 
>>- **note** : \
    &nbsp;*1.* Bert (refers to **B**idirectional **E**ncoder **R**epresentations from **T**ransformers) is base on ELMo and GPT. ELMo use RNN but Bert is based on Transformer which means ELMo needs more modifications of architecture on the subtask, but only small adjust on the top-level is needed for Bert(same as GPT), as well as, GPT is unidirectional, Bert can process the text in left-to-right and right-to-left by introduce MLM(masked LM).
    &nbsp;*2.* Bert is a pre-trained model, it can be transfer to the down-stream task with fine-tuning
    &nbsp;*3.* Firstly, pretraining ono the large unlabeled dataset, secondly, fine-tuning on a specific labeled dataset.
    &nbsp;*4.* There are two task in pre-taining: task 1: MLM, task 2:NSP(next sentence): determine whether two sentences are randomly sampled or adjacent to the original text, and learn sentence-level information, the input is a **sentence pair** so both task can help Bert learn bidirectional information.
>---
>>- **index** : 8
>>- **date** : 24/04/2024
>>- **summary** : \
    &nbsp;*1.* learning the Fakeddit dataset \
    &nbsp;*2.* learning Bert\
    &nbsp;*3.* coding
>>- **note** : \
>---
>>- **index** : 9
>>- **date** : 25/04/2024
>>- **summary** : \
    &nbsp;*1.* fine-tune Bert on Fakeddit set
>>- **note** : \
>---
>>- **index** : 10
>>- **date** : 26/04/2024
>>- **summary** : \
    &nbsp;*1.* fine-tune Bert on Fakeddit set \
    &nbsp;*2.* learn to use HPC
>>- **note** : \
>>- &nbsp;*1.* 1 epoch, acc 88.56%
>---
>>- **index** : 11
>>- **date** : 28/04/2024
>>- **summary** : \
    &nbsp;*1.* learn to use HPC \
    &nbsp;*2.* download the image set
>>- **note** : \
>>- &nbsp;*1.* 
>---
>>- **index** : 12
>>- **date** : 30/04/2024
>>- **summary** : \
    &nbsp;*1.* train the model with textual set on Bert
>>- **note** : \
>>- &nbsp;*1.accuracy:* 2-way:88.01%, 3-way: 87.91%, 6-way: 81.13% 




