parts="105"

/home/spark-3.3.4-bin-hadoop3/bin/spark-submit --conf spark.ui.showConsoleProgress=true --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=/mnt/data/tmp/spark-events task2.py "$parts"
