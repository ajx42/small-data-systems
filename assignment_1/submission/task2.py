from pyspark.sql import SparkSession
from operator import add
import sys


### USER CONFIGURATION ###
# Based on value passed from the run_task2.sh script
# Set number of partitions for this task.
NUM_PARTITIONS = int(sys.argv[1])

# Provide an application name. Note that results are saved in result_{APP_NAME} directory.
APP_NAME = "Task-2-Partitions-{}".format(NUM_PARTITIONS)

# DATASET SELECTION
# Set the path to the dataset being worked on. Set SMALL to true or false based on the datatset.

# Dataset 1: For `Berkeley Stanford`, uncomment the following.
# HDFS_INPUT_PATH = "hdfs://10.10.1.1:9000/web-BerkStan.txt"
# SMALL = True

# Dataset 2: For `Wikipedia`, uncomment the following.
HDFS_INPUT_PATH = "hdfs://10.10.1.1:9000/enwiki-pages-articles"
SMALL = False

# Set the following flat to print ranks to stdout. DO NOT set this for Wikipedia!
DEBUG = False

### END USER CONFIGURATION ###



# OUTPUT PATH
HDFS_OUTPUT_PATH = "hdfs://10.10.1.1:9000/result_{}".format(APP_NAME)

spark = SparkSession.builder.appName(APP_NAME).master("spark://c220g5-111023vm-1.wisc.cloudlab.us:7077").getOrCreate()
print(spark.sparkContext.getConf().getAll())

if SMALL:
    textDf = spark.read.text(HDFS_INPUT_PATH)
    # Berkeley-Stanford input files have comments. We need to filter them out.
    filteredDf = textDf.filter(~textDf['value'].startswith('#'))
    # Convert to edge list.
    procData = filteredDf.rdd.map(lambda x: x['value'].split('\t'))
else:
    inpData = spark.sparkContext.textFile(HDFS_INPUT_PATH)
    procData = inpData.map(lambda x: x.split('\t'))

procDataInt = procData.map(lambda x: (x[0], x[1]))

# Build Adjacency List representation of the Graph from edge list.
grpData = procDataInt.groupByKey()

# Note the above RDD does not adj lists for nodes that don't have any outgoing edges.
# This is taken care of by the fullOuterJoin in the loop.


### Partitioning Scheme
def partitioner(key):
    """Custom partitioner function"""
    return hash(key) % NUM_PARTITIONS

links = grpData.partitionBy(NUM_PARTITIONS, partitioner)
ranks = links.map(lambda x: (x[0], 1.))
### End Partitioning Scheme

# reference: https://cocalc.com/share/public_paths/960fae18301f8e8cb29472dc3ff5d0acf5659ec8/data-analysis%2Fspark-pagerank.ipynb
def computeContribs(node_urls_rank):
    """
    This function takes elements from the joined dataset above and
    computes the contribution to each outgoing link based on the
    current rank.
    """
    _, (urls, rank) = node_urls_rank
    nb_urls = len(urls)
    for url in urls:
        yield url, rank / nb_urls

for iteration in range(1):
    # compute contributions of each node where it links to
    contribs = links.join(ranks).flatMap(computeContribs)

    # use a full outer join to make sure, that not well connected nodes aren't dropped
    contribs = links.fullOuterJoin(contribs).mapValues(lambda x : x[1] or 0.0)

    # Sum up all contributions per link
    ranks = contribs.reduceByKey(add)

    # Re-calculate URL ranks
    ranks = ranks.mapValues(lambda rank: rank * 0.85 + 0.15)

# Collects all URL ranks
if DEBUG:
    for (link, rank) in sorted(ranks.collect()):
        print("%s has rank: %s." % (link, rank / total_pages))

# Print to file?
print('Num Partitions for Write: {}'.format(ranks.getNumPartitions()))
ranks.saveAsTextFile(HDFS_OUTPUT_PATH)

