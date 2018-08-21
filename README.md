# TAXI: a Taxonomy Induction Method based on Lexico-Syntactic Patterns, Substrings and Focused Crawling

This page contains implementation of a method for taxonomy induction that reached the first place in the SemEval 2016 challenge on taxonomy extraction evaluation. The method builds a taxonomy from a domain vocabulary. It extracts hypernyms from substrings and large domain-specific corpora bootstrapped from the input vocabulary. Multiple evaluations based on the SemEval taxonomy extraction datasets of four languages and three domains show state-of-the-art performance of our approach. This page contains implementations of the method including all resources needed to reproduce experiment described in the [following paper](https://www.aclweb.org/anthology/S16-1206) presented in San Diego at SemEval co-located with the NAACL'2016:

```
@inproceedings{panchenko2016taxi,
  title={TAXI at SemEval-2016 Task 13: a Taxonomy Induction Method based on Lexico-Syntactic Patterns,  Substrings and Focused Crawling},
  author={Panchenko, Alexander and Faralli, Stefano and  Ruppert, Eugen and Remus, Steffen and  Naets, Hubert and  Fairon, Cedrick and Ponzetto, Simone Paolo and Biemann, Chris},
  booktitle={Proceedings of the 10th International Workshop on Semantic Evaluation},
  year={2016},
  address={San Diego, CA, USA},
  organization={Association for Computational Linguistics}
}
```

If you would like to refer to the system please use this citation. More information about the approach can be found at [the TAXI web site](http://tudarmstadt-lt.github.io/taxi).

![taxi](https://www.lt.informatik.tu-darmstadt.de/fileadmin/_processed_/csm_taxi_662272e466.jpg)

# System Requirements

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

3. Install dependencies for using **pygraphviz**:  
  ```
  $ sudo apt-get install python-dev graphviz libgraphviz-dev pkg-config
  ```

4. Install project dependencies:

  ```
  pip install -r requirements.txt
  ```

5. Setup spaCy. Download the language models for English, Dutch, French and Italian
  ```
  $ python -m spacy download en
  $ python -m spacy download nl
  $ python -m spacy download fr
  $ python -m spacy download it
  ```

6. Setup NLTK
  ```
  $ python -m nltk.downloader stopwords
  $ python -m nltk.downloader wordnet
  ```

# Induction of SemEval Taxonomies

Run the ```semeval.py``` to reproduce experimental results, e.g.:

For a test run (few resources loaded, quick):
```
python semeval.py vocabularies/science_en.csv en simple --test
```

For a normal run (all resources are loaded, requires 64Gb of RAM):
```
python semeval.py vocabularies/science_en.csv en simple

```
Afterwards a noisy graph is being created. Clean the output by executing(this example uses the inputfile science_en.csv-relations.csv-taxo-knn1.csv): 
```
./run.sh taxi_output/simple_full/science_en.csv-relations.csv-taxo-knn1.csv

```
   
The ```vocabularies``` directory contains input terms for different domains and languages. The script lets you reproduce results in the SemEval 2016 Task 13 [Taxonomy Extraction Evaluation](http://alt.qcri.org/semeval2016/task13/) described in the [our paper](https://pdfs.semanticscholar.org/5719/932d8c194439dd08403bdb9df5ee30826e87.pdf). This script load hypernyms from the downloaded resources and constructs a taxonomy for every input vocabulary of the SemEval datasets, e.g. English Food domain. Generally, the TAXI approach takes as input a vocabulary and outputs a taxonomy for a linked subset of the terms from this vocabulary. Currently the main purpose of this repository is to ensure reproducibility of the SemEval results. The results taxonomies will be generated next to the corresponding input vocabulary file. If you need to adapt the script for your needs and require help do not hesitate to contact us.  


# Distributional Semantics

1. Download the required embeddings:  
  - `$ wget http://ltdata1.informatik.uni-hamburg.de/taxi/embeddings/embeddings_poincare_wordnet`
  - `$ wget http://ltdata1.informatik.uni-hamburg.de/taxi/embeddings/own_embeddings_w2v`
  - `$ wget http://ltdata1.informatik.uni-hamburg.de/taxi/embeddings/own_embeddings_w2v.trainables.syn1neg.npy`
  - `$ wget http://ltdata1.informatik.uni-hamburg.de/taxi/embeddings/own_embeddings_w2v.wv.vectors.npy`

2. Set the directory path in **line 45** of `distributional_semantics.py` to the directory containing the embeddings download above.

3. To apply distributional semantics to the generated taxonomy, use the script **distributional_semantics.py** or the notebook **distributional_semantics.py.ipynb**

The script can be used with following options:

| Option | Alternate | Description | Default  | Choices  |
|--------|-------------|---|---|---|
| --taxonomy | -t | Input file containing the taxonomy | - | - |
| --mode | -m | Mode of the algorithm | ds  | ds, root, remove |
| --domain | -d | Domain of the taxonomy | science | science, science_wordnet, food, environment_eurovoc |
| --exparent | -ep | Exculde parent while calculating cluster similarity | False | - |
| --exfamily | -ef | Exculde family while calculating cluster similarity | False | - |

*Example:*  
`$ python distributional_semantics.py -t taxi_output/simple_full/science_en.csv-relations.csv-taxo-knn1.csv -d food -ep`


# Visualizing taxonomies
To visualize the taxonomy structures in a **.csv** file, you must have **Networkx** and **Pygraphviz** setup in your environment.  

To construct a hierarchical taxonomy structure:  
`$ python visualize_taxonomy.py --file <csv filename>`

The images generated will be very large, so alternatively, the graph can be constructed inside the notebook **networkx_graph.ipynb**
