from pyspark.sql import SparkSession
from operator import add



### USER CONFIGURATION ###

# Provide an application name. Note that results are saved in result_{APP_NAME} directory.
APP_NAME = "Task-1"

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

# OUTPUT PATH: Generally a good idea to have the application name in the output directory name.
HDFS_OUTPUT_PATH = "hdfs://10.10.1.1:9000/result_{}".format(APP_NAME)

SPARK_MASTER = "spark://c220g5-111023vm-1.wisc.cloudlab.us:7077"
### END USER CONFIGURATION ###



spark = SparkSession.builder.appName(APP_NAME).master(SPARK_MASTER).getOrCreate()

if SMALL:
    textDf = spark.read.text(HDFS_INPUT_PATH)
    # Berkeley-Stanford input files have comments. We need to filter them out.
    filteredDf = textDf.filter(~textDf['value'].startswith('#'))
    # Convert to edge list.
    procData = filteredDf.rdd.map(lambda x: x['value'].split('\t'))
else:
    inpData = spark.sparkContext.textFile(HDFS_INPUT_PATH, 10)
    # Convert to edge list.
    procData = inpData.map(lambda x: x.split('\t'))

procDataInt = procData.map(lambda x: (x[0], x[1]))

# Build Adjacency List representation of the Graph from edge list.
grpData = procDataInt.groupByKey()

# Note the above RDD does not adj lists for nodes that don't have any outgoing edges.
# This is taken care of by the fullOuterJoin in the loop.
links = grpData
ranks = grpData.map(lambda x: (x[0], 1.))


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

for iteration in range(10):
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

# Print to file
ranks.saveAsTextFile(HDFS_OUTPUT_PATH)

