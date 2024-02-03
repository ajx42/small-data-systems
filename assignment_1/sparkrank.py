hdfs_file_path = "hdfs://10.10.1.1:9000/web-BerkStan.txt"
textDf = spark.read.text(hdfs_file_path)

# remove comments
filteredDf = textDf.filter(~textDf['value'].startswith('#'))

# edge list
procData = filteredDf.rdd.map(lambda x: x['value'].split('\t'))
procDataInt = procData.map(lambda x: (int(x[0]), int(x[1])))

# get list of all vertices
uniq = procDataInt.flatMap(lambda x: [x[0], x[1]]).distinct()
uniqData = uniq.map(lambda x: (x, 0))

# adj list from edge list
grpData = procDataInt.groupByKey()
# this is needed to have empty adj lists for vertices with no outgoing edges
grpDataFull = grpData.fullOuterJoin(uniqData)
grpDataRed = grpDataFull.map(lambda x: (x[0], x[1][0] if x[1][0] is not None else ()))

links = grpDataRed
ranks = grpDataRed.map(lambda x: (x[0], 1.))

total_pages = ranks.count()

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

from operator import add

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
for (link, rank) in sorted(ranks.collect()):
    print("%s has rank: %s." % (link, rank / len(total_pages)))

