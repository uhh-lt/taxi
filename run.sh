

CYCLE_REMOVING_TOOL="graph_pruning/graph_pruning.py"

CLEANING_TOOL="jnt/isas/cleaning.py"

EVAL_TOOL="eval/taxi_eval_archive/TExEval.jar"
EVAL_GOLD_STANDARD="eval/taxi_eval_archive/input/gold.taxo"
EVAL_ROOT=science
EVAL_JVM="-Xmx9000m"

OUTPUT_DIR=out

FILE_INPUT=$(basename "$1")
FILE_PRUNED_OUT=${FILE_INPUT}-pruned.csv
FILE_CLEANED_OUT=${FILE_PRUNED_OUT}-cleaned.csv

FILE_EVAL_TOOL_RESULT=${FILE_CLEANED_OUT}-evalresul.txt


echo Reading input: $1
echo Reading file: $FILE_INPUT
echo Output directory: $OUTPUT_DIR
echo

echo "======================================================================================================================"
echo "Cycle removing: python $CYCLE_REMOVING_TOOL $1 $OUTPUT_DIR/$FILE_PRUNED_OUT tarjan"
python $CYCLE_REMOVING_TOOL $1 $OUTPUT_DIR/$FILE_PRUNED_OUT tarjan
echo "Cycle removing finished. Written to: $OUTPUT_DIR/$FILE_PRUNED_OUT"
echo


echo "======================================================================================================================"
echo "Cleaning: python $CLEANING_TOOL $OUTPUT_DIR/$FILE_PRUNED_OUT $OUTPUT_DIR/$FILE_CLEANED_OUT $EVAL_ROOT"
python $CLEANING_TOOL $OUTPUT_DIR/$FILE_PRUNED_OUT $OUTPUT_DIR/$FILE_CLEANED_OUT $EVAL_ROOT
echo "Finished cleaning. Write output to: $OUTPUT_DIR/$FILE_CLEANED_OUT"
echo


echo "======================================================================================================================"
echo "Running eval-tool: java $EVAL_JVM -jar $EVAL_TOOL $OUTPUT_DIR/$FILE_CLEANED_OUT $EVAL_GOLD_STANDARD $EVAL_ROOT $OUTPUT_DIR/$FILE_EVAL_TOOL_RESULT"
java $EVAL_JVM -jar $EVAL_TOOL $OUTPUT_DIR/$FILE_CLEANED_OUT $EVAL_GOLD_STANDARD $EVAL_ROOT $OUTPUT_DIR/$FILE_EVAL_TOOL_RESULT
echo "Result of eval-tool written to: $OUTPUT_DIR/$FILE_EVAL_TOOL_RESULT"
echo

L_GOLD="$(wc -l $EVAL_GOLD_STANDARD | grep -o -E '^[0-9]+').0"
L_INPUT="$(wc -l $1 | grep -o -E '^[0-9]+').0"
RECALL="$(tail -n 1 $OUTPUT_DIR/$FILE_EVAL_TOOL_RESULT | grep -o -E '[0-9]+[\.]?[0-9]*')"


PRECISION=$(echo "print $RECALL * $L_GOLD / $L_INPUT" | python)
F1=$(echo "print 2 * $RECALL * $PRECISION / ($PRECISION + $RECALL)" | python)

echo "Recall: $RECALL"
echo "Precision: $PRECISION"
echo "F1: $F1"


echo "Script finished."






