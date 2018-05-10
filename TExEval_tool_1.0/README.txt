================================================================================
                              SEMEVAL-2015 TASK 17
                   TExEval: Taxonomy Extraction Evaluation
	Paul Buitelaar, Georgeta Bordea, Roberto Navigli and Stefano Faralli
================================================================================
			http://alt.qcri.org/semeval2015/task17/
================================================================================
TExEval_tool_1.0
================================================================================

==================
PACKAGE CONTENTS
==================

The tool package contains the following:

README.txt                                        This file
TExEval.jar		                          Taxonomy evaluation tool used for the TExEval shared task
runExamplePrecision.sh			          Linux script for running the example evaluation by computing Precision using human judgements
runExampleVSGoldStandard.sh			  Linux script for running the example evaluation using Gold Standard comparison
example/gold1.taxo				  Example gold standard taxonomy
example/sys1.taxo				  Example taxonomy produced by a system 
example/sys1.taxo.eval				  Example of human judgements for relations from the above taxonomy
example/results.txt				  Example output of the taxonomy evaluation tool system

=============
INPUT FORMAT
=============

The format of the input files for the system and gold standard taxonomies is a
tab-separated fields:

relation_id <TAB> term <TAB> hypernym 

where:
- relation_id: is a relation identifier; 
- term: is a term of the taxonomy;
- hypernym: is a hypernym for the term. 

e.g

0<TAB>cat<TAB>animal
1<TAB>dog<TAB>animal
2<TAB>car<TAB>animal
....

The format of the input files with human judgements of taxonomy relations is a
tab-separated fields:

relation_id <TAB> eval

where:
- relation_id: is a relation identifier; 
- eval: is an empty string if the relation is good, an "x" otherwise

e.g.
0<TAB>
1<TAB>
2<TAB>x
....

==========================
  EVALUATION METRICS
==========================

The TExEval.jar is a runnable jar, which compares a system-generated taxonomy against a gold standard taxonomy. The measures reported by the program are:
1) A measure to compare the overall structure of the taxonomy against a gold standard, with an approach used for comparing hierarchical clusters[¹];
2) Precision: the number of correct relations over the number of given relations;
3) Recall: the number of relation in common with the gold standard over the number of gold standard relations; 



==========================
TExEval.jar usage
==========================
To run TExEval.jar on your linux machine, open a terminal and enter:
"java -jar TExEval.jar system.taxo goldstandard.taxo root results"
or
"java -jar TExEval.jar system.taxo.eval results"

where:
- system.taxo: is the taxonomy produced by your system;
- system.taxo.eval: is the evaluation of the system produced relations;
- goldstandard.taxo: is the gold standard taxonomy;
- root: is the common root node for the system and the goldstandard taxonomies
- result: is the file where the program will write the results.

By running the runExampleVSGoldStandard.sh, the TExEval.jar compare the following system produced taxonomy:

example/sys1.taxo
0	a	entity
1	b	a
2	c	b
3	d	b
4	e	b

against the following gold standard taxonomy:

example/gold1.taxo
0	a	entity
1	b	a
2	c	b
3	d	b
4	e	b
5	f	e
6	g	e
7	h	e

producing the following result.txt file

example/rsults.txt
Taxonomy file	./example/sys1.taxo
Gold Standard file	./example/gold1.taxo
Root	entity
level	B	Weight	BxWeight
0	0.18257418583505536	1.0	0.18257418583505536
1	0.18257418583505536	0.5	0.09128709291752768
2	0.18257418583505536	0.3333333333333333	0.06085806194501845
3	0.0	0.25	0.0
Cumulative Measure	0.16066528353484874
Recall from relation overlap	0.625

where:
1) the first two lines report the arguments passed to the jar application
2) a structural comparison of the system taxonomy against the gold standard taxonomy[¹] 
3) the estimated Recall


By running the runExamplePrecision.sh, the TExEval.jar compute the Precision from the following Evaluation file for the system produced relation:

example/sys1.taxo.eval
0	
1	
2	
3	
4	x

and produce the following result.txt file:

Taxonomy relation evaluation file	./example/sys1.taxo.eval
Precision from relation evaluation	0.8



[¹] Paola Velardi, Stefano Faralli, Roberto Navigli. OntoLearn Reloaded: A Graph-based Algorithm for Taxonomy Induction. Computational Linguistics, 39(3), MIT Press, 2013, pp. 665-707.

