BIN="../../build/experiments/splitDataMovement"
OUTPUT_FILE="./data.csv"
HEADER="kind,split,size(Byte),time(s),speed(GB/s)"

echo $HEADER > $OUTPUT_FILE


$BIN -no-header -start-size=100000 -end-size=8000000 -step-size=100000 -split=1 >> $OUTPUT_FILE
$BIN -no-header -start-size=100000 -end-size=8000000 -step-size=100000 -split=2 >> $OUTPUT_FILE
$BIN -no-header -start-size=100000 -end-size=8000000 -step-size=100000 -split=4 >> $OUTPUT_FILE
$BIN -no-header -start-size=100000 -end-size=8000000 -step-size=100000 -split=8 >> $OUTPUT_FILE

$BIN -no-header -use-unified-memory -start-size=1000000 -end-size=80000000 -step-size=1000000 -split=1 >> $OUTPUT_FILE
$BIN -no-header -use-unified-memory -start-size=1000000 -end-size=80000000 -step-size=1000000 -split=2 >> $OUTPUT_FILE
$BIN -no-header -use-unified-memory -start-size=1000000 -end-size=80000000 -step-size=1000000 -split=4 >> $OUTPUT_FILE
$BIN -no-header -use-unified-memory -start-size=1000000 -end-size=80000000 -step-size=1000000 -split=8 >> $OUTPUT_FILE
