# TAXI
TAXI: a Taxonomy Induction Method based on Lexico-Syntactic Patterns, Substrings and Focused Crawling

More information about the approach can be found at [the TAXI web site](http://tudarmstadt-lt.github.io/taxi).

# System requirements

The system was tested on Debian/Ubuntu Linux and Mac OS X. To load all resources in memory you need about 64 Gb of RAM. 

# Installation 

1. Clone repository: 

  ```
  git clone https://github.com/tudarmstadt-lt/taxi.git
  ```

2. Download resources into the repository (4.4G compressed by gzip):

  ```
  cd taxi && wget http://panchenko.me/data/joint/taxi/res/resources.tgz && tar xzf resources.tgz
  ```

3. Install dependencies:

  ```
  pip install -r requirements.txt
  ```

4. Run the ```semeval.py``` to reproduce experimental results, e.g.:
    ```
    python semeval.py vocabularies/science_en.csv en simple
    ```
    The ```vocabularies``` directory contains input terms for different domains and languages. The script lets you reproduce results in the SemEval 2016 Task 13 [Taxonomy Extraction Evaluation](http://alt.qcri.org/semeval2016/task13/) described in the [our paper](https://pdfs.semanticscholar.org/5719/932d8c194439dd08403bdb9df5ee30826e87.pdf). This script load hypernyms from the downloaded resources and constructs a taxonomy for every input vocabulary of the SemEval datasets, e.g. English Food domain. Generally, the TAXI approach takes as input a vocabulary and outputs a taxonomy for a linked subset of the terms from this vocabulary. Currently the main purpose of this repository is to ensure reproducibility of the SemEval results. The results taxonomies will be generated next to the corresponding input vocabulary file. If you need to adapt the script for your needs and require help do not hesitate to contact us.  
