import urllib.request

import sys


def optimize_solr_index(solr_url, corename):
    code = urllib.request. \
        urlopen("{}/{}/update?optimize=true".
                format(solr_url, corename)).read()


if __name__=="__main__":
    optimize_solr_index(sys.argv[1],sys.argv[2])