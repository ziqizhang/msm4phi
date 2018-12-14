package uk.ac.shef.inf.msm4phi.app;

import org.apache.solr.client.solrj.SolrClient;
import uk.ac.shef.inf.msm4phi.IndexAnalyserMaster;
import uk.ac.shef.inf.msm4phi.Util;
import uk.ac.shef.inf.msm4phi.export.WorkerTweetExportByDays;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

//todo: export a sample of X tweets per Y days, can define params such as length of tweets
public class AppGenericExporterByDays {
    public static void main(String[] args) throws IOException {
        SolrClient solrClient = Util.getSolrClient(Paths.get("/home/zz/Work/msm4phi/resources/solr_offline"),"tweets");
        WorkerTweetExportByDays worker = new WorkerTweetExportByDays(0,solrClient,
                "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/nodexl",
                1, 0.1);
        IndexAnalyserMaster exporter=new IndexAnalyserMaster(
                new File("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/2_PART2_processed_hashtags.tsv"),
                worker
        );
        exporter.setThreads(1);
        exporter.process();
        solrClient.close();
        System.exit(0);
    }
}
