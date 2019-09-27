This page contains implementation of a method for taxonomy induction that reached the first place in the [SemEval 2016 challenge on  taxonomy extraction evaluation](http://alt.qcri.org/semeval2016/task13/). The method builds a taxonomy from a domain vocabulary. It extracts hypernyms from substrings and large domain-specific corpora bootstrapped from the input vocabulary. Multiple evaluations based on the SemEval taxonomy extraction datasets of four languages and three domains show state-of-the-art performance of our approach. This page contains implementations of the method including all resources needed to reproduce experiment described in the [following paper](http://web.informatik.uni-mannheim.de/ponzetto/pubs/panchenko16.pdf): 

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
If you would like to refer to the system please use this citation. 

### Motivation
TAXI is a taxonomy induction method first presented at the SemEval 2016 challenge on Taxonomy Extraction Evaluation. We consider taxonomy induction as a process that should -- as much as possible -- be driven solely on the basis of raw text processing. While some labeled examples might be utilized to tune the extraction and induction process, we avoid relying on structured lexical resources such as WordNet or BabelNet. We rather envision a situation where a taxonomy shall be induced in a new domain or a new language for which such resources do not exist. Otherwise, there is little need for induction,  and in application-based scenarios it is still possible to merge induced and existing taxonomies. In this paper, we demonstrate our methodology by executing hyponymy pattern extraction on general-domain and domain-specific corpora for four languages.

![TAXI](https://upload.wikimedia.org/wikipedia/commons/3/37/Car_rapide.jpg)

### Taxonomy Induction Method

Our approach is characterized by scalability and simplicity, assuming that being able to process larger input data is more important than the complexity of the approach. Our approach to taxonomy induction takes as input a set of domain terms and general-domain text corpora and outputs a taxonomy. It consist of four steps. First, we crawl domain-specific corpora based on terminology of the target domain. These compliment general purpose corpora, like Wikipedia. Second, candidate hypernyms are extracted based on substrings and lexico-syntactic patterns. These candidates are subsequently pruned so that each term has only few most salient hypernyms. The last step performs optimization of the overall taxonomy structure removing cycles and linking disconnected components to the root. Below we present a description of each of these steps. Full description of the method is available in [our SemEval paper](http://semeval-paper.com).

### Download Resources
- [Input of the TAXI](http://panchenko.me/data/joint/taxi/terms/): domain vocabularies of the three domains (Food, Science and Environment)
- [Output of the TAXI](http://panchenko.me/data/joint/taxi/release-final/): taxonomies of the three domains submitted to SemEval 2016 Task 13
- [Resources used by TAXI](http://panchenko.me/data/joint/taxi/res/resources/): collections of extracted hypernyms for English, French, Italian and Dutch for the three domains (Food, Science and Environment). Below you can find separate collections of hypernyms for each language domain-pair, where language is English, French, Dutch or Italian and domain is Environment, Food or Science. The following table sizes of these hypernym relation databases (see more details in the original publication mentioned above). 

  * [English General Domain Hypernyms (WebISA)](http://panchenko.me/data/joint/taxi/res/resources/en_cc.csv.gz)
  * [English General Domain Hypernyms (PattenSim)](http://panchenko.me/data/joint/taxi/res/resources/en_ps.csv.gz)
  * [English General Domain Hypernyms (PattaMaika)](http://panchenko.me/data/joint/taxi/res/resources/en_pm.csv.gz)
  * [English Environment Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/en_environment.csv.gz)
  * [English Food Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/en_food.csv.gz)
  * [English Science Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/en_science.csv.gz)
  * [French General Domain Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/fr.csv.gz)
  * [French Environment Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/fr_environment.csv.gz)
  * [French Food Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/fr_food.csv.gz)
  * [French Science Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/fr_science.csv.gz)
  * [Dutch General Domain Hypernyms (PattaMaika)](http://panchenko.me/data/joint/taxi/res/resources/nl.csv.gz)
  * [Dutch Environment Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/nl_environment.csv.gz)
  * [Dutch Food Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/nl_food.csv.gz)
  * [Dutch Science Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/nl_science.csv.gz)
  * [Italian General Domain Hypernyms (PattaMaika)](http://panchenko.me/data/joint/taxi/res/resources/it.csv.gz)
  * [Italian Environment Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/it_environment.csv.gz)
  * [Italian Food Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/it_food.csv.gz)
  * [Italian Science Hypernyms (PatternSim)](http://panchenko.me/data/joint/taxi/res/resources/it_science.csv.gz)
  

- [Corpora used to extract hypernyms used by TAXI](http://panchenko.me/data/joint/taxi/corpora/): general collections and those gathered with the focused crawler.

### Useful Links

- [SemEval 20016, Task 13: Taxonomy Extraction Evaluation (TExEval-2)](http://alt.qcri.org/semeval2016/task13/) 
- [SemEval 2015, Task 17: Taxonomy Extraction Evaluation (TExEval)] (http://alt.qcri.org/semeval2015/task17/)
- [Language Technology Group of TU Darmstadt](https://www.lt.informatik.tu-darmstadt.de/de/lt-home/)
- [Serelex](http://www.serelex.org): a lexico-semantic search engine
- [JoBimText framework for distributional semantics](http://jobimtext.org)

### Contact
If you have any questions regarding the project write to Alexander Panchenko (email available at http://panchenko.me) or open a Github issue. 
