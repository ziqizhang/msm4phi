package uk.ac.shef.inf.msm4phi.app;

import uk.ac.shef.inf.msm4phi.MasterTweetExport;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

/**
 * This should be used only after 'favorite' and 'retweet' stats are collected as a postprocess, see AppPostProcess
 */
public class AppNodeXLExporter {
    public static void main(String[] args) throws IOException {
        MasterTweetExport exporter=new MasterTweetExport(
                Paths.get("/home/zz/Work/msm4phi/resources/solr_offline"),"tweets",
                new File("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/2_PART2_processed_hashtags.tsv"),
                "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/nodexl"
        );
        exporter.setThreads(1);
        exporter.process();
    }
}
