package uk.ac.shef.inf.msm4phi.app;

import org.apache.solr.client.solrj.SolrClient;
import uk.ac.shef.inf.msm4phi.IndexPopulatorUser;
import uk.ac.shef.inf.msm4phi.PostProcessor;
import uk.ac.shef.inf.msm4phi.Util;

import java.nio.file.Paths;

/**
 * This should be used only after 'favorite' and 'retweet' stats are collected as a postprocess, see AppPostProcess
 */
public class AppUserIndexGenerator {
    public static void main(String[] args) {
        SolrClient tweetSolrClient= Util.getSolrClient(Paths.get(args[0]),"tweets");
        SolrClient userSolrClient= Util.getSolrClient(Paths.get(args[0]),"tweets");
        IndexPopulatorUser pprocess= new IndexPopulatorUser(tweetSolrClient,userSolrClient,
                args[1],args[2],args[3],args[4]);
        pprocess.process();
    }
}
