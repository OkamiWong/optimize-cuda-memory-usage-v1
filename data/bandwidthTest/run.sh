BIN="../../build/experiments/bandwidthTest"
OUTPUT_FILE="./data.csv"
HEADER="kind,size(Byte),time(s),speed(GB/s)"

echo $HEADER > $OUTPUT_FILE

$BIN -no-header -start-size=100000 -end-size=8000000 -step-size=100000 >> $OUTPUT_FILE

$BIN -no-header -use-unified-memory -start-size=1000000 -end-size=80000000 -step-size=1000000 >> $OUTPUT_FILE
