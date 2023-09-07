BIN="../../build/experiments/splitDataMovement"
OUTPUT_FILE="./data-nvlink.csv"
HEADER="kind,split,size(Byte),time(s),speed(GB/s)"

echo $HEADER > $OUTPUT_FILE

$BIN -no-header -use-nvlink -use-log-scale -start-size=1000000 -end-size=10000000000 -step-size=2 -split=1 >> $OUTPUT_FILE
$BIN -no-header -use-nvlink -use-log-scale -start-size=1000000 -end-size=10000000000 -step-size=2 -split=2 >> $OUTPUT_FILE
$BIN -no-header -use-nvlink -use-log-scale -start-size=1000000 -end-size=10000000000 -step-size=2 -split=4 >> $OUTPUT_FILE
$BIN -no-header -use-nvlink -use-log-scale -start-size=1000000 -end-size=10000000000 -step-size=2 -split=8 >> $OUTPUT_FILE

$BIN -no-header -use-nvlink -use-unified-memory -use-log-scale -start-size=1000000 -end-size=10000000000 -step-size=2 -split=1 >> $OUTPUT_FILE
$BIN -no-header -use-nvlink -use-unified-memory -use-log-scale -start-size=1000000 -end-size=10000000000 -step-size=2 -split=2 >> $OUTPUT_FILE
$BIN -no-header -use-nvlink -use-unified-memory -use-log-scale -start-size=1000000 -end-size=10000000000 -step-size=2 -split=4 $OUTPUT_FILE
$BIN -no-header -use-nvlink -use-unified-memory -use-log-scale -start-size=1000000 -end-size=10000000000 -step-size=2 -split=8 >> $OUTPUT_FILE
