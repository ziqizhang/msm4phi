package uk.ac.shef.inf.msm4phi.feature.kim;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;
import uk.ac.shef.inf.msm4phi.Util;
import uk.ac.shef.inf.msm4phi.feature.oirbaseline.ProfileReplacer;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * exports tweets of each user to a file to be further processed to calc. cosine similarities by 'kim_feature_extractor.py'
 *
 * for each user, a text file is created, first line=user name; following lines contain a prefix: ot/rt followed by | then
 * the tweet
 */
public class UserTweetExporter {
    private static final Logger LOG = Logger.getLogger(UserTweetExporter.class.getName());

    public static void main(String[] args) throws IOException {
        SolrClient solrClient =
                Util.getSolrClient(Paths.get(args[1]), "tweets");
        UserTweetExporter ute = new UserTweetExporter();
        ute.export(14,args[0], solrClient, args[2]);
    }


    public void export(int userCol, String userProfileCSV,
                       SolrClient tweets,
                       String outFolder) throws IOException {
        Reader reader = Files.newBufferedReader(Paths.get(userProfileCSV));
        CSVReader csvReader = new CSVReader(reader);


        // Reading Records One by One in a String array
        String[] nextRecord;
        int count = 0;
        while ((nextRecord = csvReader.readNext()) != null) {
            count++;
            if (count == 1) {
                continue;
            }

            String user = nextRecord[userCol];
            PrintWriter p = new PrintWriter(outFolder+"/"+count+".txt");
            p.println(user);

            List<String> userTweets = fetch(user, tweets);
            for (String s : userTweets) {
                p.println(s);
            }

            p.close();
        }
        csvReader.close();
        System.exit(0);
    }

    /**
     * get tweets of a user
     *
     * @param user
     * @return
     */
    public List<String> fetch(String user, SolrClient tweetIndex) {
        SolrQuery q = Util.createQueryTweetsOfUserScreenname(user, 5000);
        List<String> tweets = new ArrayList<>();

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
                for (SolrDocument d : resultDocs) {
                    String text = d.getFieldValue("status_text").toString();
                    Object retweeted=d.getFieldValue("retweeted_status_id_str");
                    if (retweeted ==null || retweeted.toString().length()==0)
                        tweets.add("ot|"+text.replaceAll("\\s+"," "));
                    else
                        tweets.add("rt|"+text.replaceAll("\\s+"," "));
                }


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

        return tweets;
    }


}
