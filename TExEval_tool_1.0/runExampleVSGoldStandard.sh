#!/bin/sh
java -Xmx9000m -jar TExEval.jar ./example/sys1.taxo ./example/gold1.taxo entity ./example/resultsVSGoldStandard.txt
