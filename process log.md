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


