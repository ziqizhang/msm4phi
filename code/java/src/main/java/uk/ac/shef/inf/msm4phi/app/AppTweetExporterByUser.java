package uk.ac.shef.inf.msm4phi.app;

import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrServerException;
import uk.ac.shef.inf.msm4phi.IndexAnalyserMaster;
import uk.ac.shef.inf.msm4phi.Util;
import uk.ac.shef.inf.msm4phi.analysis.UserFeatureExporter;
import uk.ac.shef.inf.msm4phi.export.WorkerTweetExportByDays;
import uk.ac.shef.inf.msm4phi.export.WorkerTweetExportByUserType;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

public class AppTweetExporterByUser {
    public static void main(String[] args) throws IOException, SolrServerException {
        SolrClient userIndex =
                Util.getSolrClient(Paths.get(args[0]), "users");
        SolrClient tweetIndex =
                Util.getSolrClient(Paths.get(args[0]), "tweets");
        WorkerTweetExportByUserType worker =
                new WorkerTweetExportByUserType(0,userIndex, tweetIndex,
                args[1],
                "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/nodexl");
        IndexAnalyserMaster exporter=new IndexAnalyserMaster(
                new File("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/2_PART2_processed_hashtags.tsv"),
                worker
        );
        userIndex.close();
        tweetIndex.close();
        System.exit(0);
    }

}
