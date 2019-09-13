package uk.ac.shef.inf.msm4phi.feature;

import com.opencsv.CSVReader;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocumentList;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * finds tweets posted by/not by a list of users
 */

public class UserTweetsFetcher {
    private SolrClient userIndex;
    private SolrClient tweetIndex;
    private String userProfileCSV;

    private static final Logger LOG = Logger.getLogger(UserTweetsFetcher.class.getName());

    public UserTweetsFetcher(int id,
                             SolrClient tweetIndex,
                             String userProifleCSV) {
        this.tweetIndex = tweetIndex;
        this.userProfileCSV=userProifleCSV;
    }

    public static void main(String[] args) throws IOException {
        SolrClient solrClient =
                Util.getSolrClient(Paths.get(args[0]), "tweets");
        //read training data containing users and profiles
        UserTweetsFetcher fetcher= new UserTweetsFetcher(1, solrClient, args[1]);
        Map<String, String> userProfiles= fetcher.readUserProfileCSV(14,22);

        //check tweets
        Set<String> users = userProfiles.keySet();
        long total=0, noTweets=0, userCount=0;

        for (String u : users){
            userCount++;
            System.out.println(userCount);
            long res=fetcher.fetch(u);
            total+=res;
            if (res==0) {
                noTweets++;
                String profile=userProfiles.get(u);
                System.out.println();
            }
        }

        System.out.println(total);
        System.out.println(noTweets);
        System.exit(0);
    }

    public Map<String, String> readUserProfileCSV(int userCol, int profileCol) throws IOException {
        Map<String, String> profiles=new HashMap<>();
        Reader reader = Files.newBufferedReader(Paths.get(userProfileCSV));
        CSVReader csvReader = new CSVReader(reader);

        // Reading Records One by One in a String array
        String[] nextRecord;
        int count=0;
        while ((nextRecord = csvReader.readNext()) != null) {
            if (count==0){
                count++;
                continue;
            }
            profiles.put(nextRecord[userCol], nextRecord[profileCol]);
        }
        return profiles;
    }

    /**
     * Query the solr backend to process tweets
     *
     * @return
     */
    protected long fetch(String user) {

        SolrQuery q = Util.createQueryTweetsOfUserScreenname(user, 5000);

        boolean stop = false;
        long total = 0;
        while (!stop) {
            QueryResponse res = null;

            try {
                res = Util.performQuery(q, tweetIndex);
                if (res != null)
                    total = res.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal tweets for %s, of %d, currently processing from %d to %d...",
                        user, total, q.getStart(), q.getStart() + q.getRows()));
                SolrDocumentList resultDocs = res.getResults();


            } catch (Exception e) {
                LOG.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                        q.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
            }

            int curr = q.getStart() + q.getRows();
            if (curr < total)
                q.setStart(curr);
            else
                stop = true;
        }

        return total;
    }
}

