#!/bin/bash
  
${SPARK_HOME_DIR}/bin/spark-submit --conf spark.ui.showConsoleProgress=true --conf spark.eventLog.enabled=true part2.py ${IN} ${OUT}
