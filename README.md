# FeedbackBot
A repo of the code used for a consulting project at the Summer 2018 Insight Data Science Fellowship, to generate a FeedbackBot. There are several files that are excluded from this repo for various reasons that are described below in the section entitled "Missing Files".


## Missing files:
Notably missing from this document:
* file/fasttext_model.bin 
  * this is a FastText model that was trained on student responses. There are pre-trained English models that can be used in it's place (though, note that the pretrained models often require much more storage and memory than smaller files, and also assume correct spellings. If you would like to account for made up words or misspellings it is better to train on input that has misspellings). Also note that ay fasttext model trained in C++ is incompatible with the python wrapper.
* file/stuff.json
  * this is a json file that contains
