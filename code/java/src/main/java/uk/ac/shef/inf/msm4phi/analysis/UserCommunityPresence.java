package uk.ac.shef.inf.msm4phi.analysis;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

/**
 * calculate:
 *
 * -% of users that are found in >1 communities
 * -for the above hashtags, calculate max, min, 75%, 25%, 50%, median
 * -print top 100 users and their community numbers
 *
 *
 */

public class UserCommunityPresence {

    private static final int resultBatchSize = 10000;
    private static final Logger LOG = Logger.getLogger(UserCommunityPresence.class.getName());

    void process(String hashtagFile, String outFile,
                 SolrClient userCore) throws IOException {
        Map<String, String> tag2diseaseInput = Common.createInverseHashtagMap(hashtagFile);
        LOG.info("Calculating users in multiple disease communities...");
        Pair<Map<String, Set<String>>, Map<String, String>> res
                = Common.findDiseasesAndUsers(userCore, tag2diseaseInput, LOG, resultBatchSize);

        //users found in at least 2 communities
        Map<String, Set<String>> selected = new HashMap<>();
        List<Double> diseaseNum = new ArrayList<>();
        for(Map.Entry<String, Set<String>> en:res.getLeft().entrySet()){
            if(en.getValue().size()>1){
                selected.put(en.getKey(), en.getValue());
                diseaseNum.add((double)en.getValue().size());
            }
        }

        Collections.sort(diseaseNum);
        Collections.reverse(diseaseNum);
        List<String> users = new ArrayList<>(selected.keySet());
        users.sort((s, t1) ->
                Integer.compare(selected.get(t1).size(), selected.get(s).size()));
        double[] values = ArrayUtils.toPrimitive(diseaseNum.toArray(new Double[0]));


        PrintWriter p = new PrintWriter(outFile);
        StringBuilder sb = new StringBuilder();
        sb.append("total unique users,").append(res.getLeft().size()).append("\n")
                .append("multi-community presence as %,").append((double)selected.size()/res.getLeft().size()).append("\n")
                .append("max community presence,").append(StatUtils.max(values)).append("\n")
                .append(".75 quantile,").append(StatUtils.percentile(values, 0.75)).append("\n")
                .append(".50 quantile (median),").append(StatUtils.percentile(values, 0.50)).append("\n")
                .append(".25 quantile,").append(StatUtils.percentile(values, 0.25)).append("\n")
                .append("min community presence,").append(StatUtils.min(values)).append("\n")
                .append("average,").append(StatUtils.mean(values)).append("\n\n")
                .append("Top 100 by community presence (as #of communities)\n");

        for (int i=0; i<500; i++){
            sb.append(res.getRight().get(users.get(i))).append(",").append(diseaseNum.get(i)).append("\n");
        }
        sb.append("\n");

        List<List<Double>> outliers= Util.detectOutliersIQR(values);
        sb.append("OUTLIERS,").append(outliers.get(1).size()).append("\n");
        for (int i=0; i<outliers.get(1).size(); i++){
            sb.append(res.getRight().get(users.get(i))).append(",").append(diseaseNum.get(i)).append("\n");
        }

        p.println(sb.toString());
        p.close();


    }
}
