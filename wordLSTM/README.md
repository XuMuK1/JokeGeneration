# Word Level LSTM 

This model is partially inherited from pytorch examples. I completely changed preprocessing module, made some technical fixes in training and added new logging procedures. All charts are made with matplotlib and can be found in ipynb and pics section.

## Short overview

The model trains on plain text files. You need to form three text files train.txt,test.txt,valid.txt,all.txt. The first one contains training part of the texts, valid.txt and test.txt are used for validation and final testing, all.txt consits of all contents of these three files and used to train the dictionary.

To run model, use

```{bash}
python generate.py --help
```
and 
```{bash}
python main.py --help
```
to train new models.

## IMPORTANT NOTE

The training dataset originally consists of raw scraped jokes, so the quality of output jokes and politeness varies greatly.