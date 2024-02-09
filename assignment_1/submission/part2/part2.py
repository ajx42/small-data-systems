from pyspark.sql import SparkSession
import argparse

# argument parser
parser = argparse.ArgumentParser(description='Part 2')
parser.add_argument('input_file', help='Path to the input file.')
parser.add_argument('output_file', help='Path to the output file.')

args = parser.parse_args()

APP_NAME = "part2"

# dataset
HDFS_INPUT_PATH = "hdfs://10.10.1.1:9000/{}".format(args.input_file)
HDFS_OUTPUT_PATH = "hdfs://10.10.1.1:9000/{}".format(args.output_file)

spark = (SparkSession
	.builder
        .appName(APP_NAME)
	.config("some.config.option", "some-value")
	.master("spark://10.10.1.1:7077")
	.getOrCreate())

# read data
df = spark.read.options(header=True).csv(HDFS_INPUT_PATH)

# sort data
df_sorted = df.orderBy("cca2", "timestamp")

# save data to hdfs
df_sorted.write.csv(HDFS_OUTPUT_PATH)

