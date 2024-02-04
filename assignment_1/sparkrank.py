from pyspark.sql import SparkSession
from operator import add

APP_NAME = "stan-withwrite"

# Dataset 1: Berkeley Stanford
# HDFS_INPUT_PATH = "hdfs://10.10.1.1:9000/web-BerkStan.txt"
# SMALL = True

# Dataset 2: Wikipedia
HDFS_INPUT_PATH = "hdfs://10.10.1.1:9000/enwiki-pages-articles"
SMALL = False


HDFS_OUTPUT_PATH = "hdfs://10.10.1.1:9000/result_{}".format(APP_NAME)

# DEBUG will print out calculated page ranks to stdout
# Don't use it with Wikipedia dataset.
DEBUG = False

spark = SparkSession.builder.appName(APP_NAME).master("spark://c220g5-111023vm-1.wisc.cloudlab.us:7077").getOrCreate()

if SMALL:
    textDf = spark.read.text(HDFS_INPUT_PATH)
    # remove comments
    filteredDf = textDf.filter(~textDf['value'].startswith('#'))
    # edge list
    procData = filteredDf.rdd.map(lambda x: x['value'].split('\t'))
else:
    inpData = spark.sparkContext.textFile(HDFS_INPUT_PATH, 10)
    procData = inpData.map(lambda x: x.split('\t'))

procDataInt = procData.map(lambda x: (x[0], x[1]))

# adj list from edge list
grpData = procDataInt.groupByKey()

# Note the above RDD does not adj lists for nodes that don't have any outgoing edges.
# I think this is still fine as there is a full outer join in the pagerank impl.

links = grpData
ranks = grpData.map(lambda x: (x[0], 1.))

total_pages = ranks.count()

print("total pages in this dataset: {}".format(total_pages))

# from: https://cocalc.com/share/public_paths/960fae18301f8e8cb29472dc3ff5d0acf5659ec8/data-analysis%2Fspark-pagerank.ipynb
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

# If we are not saving to file, this operation will ensure all computation has finished
# total_results = ranks.count()

# Print to file?
ranks.saveAsTextFile(HDFS_OUTPUT_PATH)

#samples = ranks.takeSample(False, 1000)
#data_to_write = "\n".join(map(str, taken_elements))
#spark.sparkContext.parallelize([data_to_write]).saveAsTextFile(HDFS_OUTPUT_PATH)


