package uk.ac.shef.inf.msm4phi.app;

import org.apache.solr.client.solrj.SolrClient;
import uk.ac.shef.inf.msm4phi.IndexAnalyserMaster;
import uk.ac.shef.inf.msm4phi.Util;
import uk.ac.shef.inf.msm4phi.export.WorkerTweetExportNodeXL;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

/**
 * This should be used only after 'favorite' and 'retweet' stats are collected as a postprocess, see AppPostProcess
 */
public class AppNodeXLExporter {
    public static void main(String[] args) throws IOException {
        SolrClient solrClient = Util.getSolrClient(Paths.get("/home/zz/Work/msm4phi_data/solr"),"tweets");
        WorkerTweetExportNodeXL worker = new WorkerTweetExportNodeXL(0,solrClient,
                "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/nodexl");
        IndexAnalyserMaster exporter=new IndexAnalyserMaster(
                new File("/home/zz/Work/msm4phi/data/symplur_hashtags/2_processed_hashtags.tsv"),
                worker
        );
        exporter.setThreads(1);
        exporter.process();
        solrClient.close();
        System.exit(0);
    }
}
