package uk.ac.shef.inf.msm4phi;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.lucene.index.Terms;
import org.apache.solr.client.solrj.SolrClient;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;

public class MasterTweetExport {

    private static final Logger LOG = Logger.getLogger(MasterTweetExport.class.getName());

    private SolrClient solrClient;
    private Map<String, List<String>> hashtagMap;
    private int threads=1;
    private String outFolder;

    public MasterTweetExport(Path solrHome, String solrCore, File hashtagFile, String outFolder) throws IOException {
        this.solrClient= Util.getSolrClient(solrHome, solrCore);
        this.hashtagMap=Util.readHashtags(hashtagFile);
        this.outFolder=outFolder;
    }

    public void process() {
        try {

            int maxPerThread = hashtagMap.size() / threads;


            LOG.info(String.format("Beginning processing %d hastags on %d threads, at %s", hashtagMap.size(), threads,
                    new Date().toString()));
            WorkerTweetExport exporter = new WorkerTweetExport(this.solrClient, this.hashtagMap, maxPerThread, outFolder);

            ForkJoinPool forkJoinPool = new ForkJoinPool(maxPerThread);
            int total = forkJoinPool.invoke(exporter);

            LOG.info(String.format("Completed $%d hashtags at %s", total, new Date().toString()));

            solrClient.close();

        } catch (Exception ioe) {
            StringBuilder sb = new StringBuilder("Failed to build features!");
            sb.append("\n").append(ExceptionUtils.getFullStackTrace(ioe));
            LOG.error(sb.toString());
        }

    }

    public void setThreads(int threads) {
        this.threads = threads;
    }
}
