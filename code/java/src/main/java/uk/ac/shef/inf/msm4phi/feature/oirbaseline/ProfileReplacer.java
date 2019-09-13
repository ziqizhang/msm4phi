package uk.ac.shef.inf.msm4phi.feature.oirbaseline;

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
import uk.ac.shef.inf.msm4phi.feature.UserTweetsFetcher;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * replaces profiles in /home/zz/Work/msm4phi_data/paper2/reported/training_data/paper_reported/basic_features_empty_profile_filled.csv
 * with a concatenation of tweets by each user, output the new csvfile
 */

public class ProfileReplacer {

    private static final Logger LOG = Logger.getLogger(ProfileReplacer.class.getName());

    public static void main(String[] args) throws IOException {
        SolrClient solrClient =
                Util.getSolrClient(Paths.get(args[1]), "tweets");
        ProfileReplacer pr = new ProfileReplacer();
        pr.replace(14,22,args[0], solrClient, args[2]);
    }


    public void replace(int userCol, int profileCol, String userProfileCSV,
                        SolrClient tweets,
                        String userProfileNewCSV) throws IOException {
        Reader reader = Files.newBufferedReader(Paths.get(userProfileCSV));
        CSVReader csvReader = new CSVReader(reader);

        Writer writer = Files.newBufferedWriter(Paths.get(userProfileNewCSV), Charset.forName("utf-8"));

        CSVWriter csvWriter = new CSVWriter(writer,
                CSVWriter.DEFAULT_SEPARATOR,
                CSVWriter.DEFAULT_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END);

        // Reading Records One by One in a String array
        String[] nextRecord;
        int count = 0;
        while ((nextRecord = csvReader.readNext()) != null) {
            if (count == 0) {
                count++;
                csvWriter.writeNext(nextRecord);
                continue;
            }

            String user = nextRecord[userCol];
            List<String> userTweets = fetch(user, tweets);
            if (userTweets.size()==0) {
                csvWriter.writeNext(nextRecord);
                continue;
            }
            StringBuilder sb = new StringBuilder();
            for (String s : userTweets) {
                sb.append(s).append(" ");
            }
            nextRecord[profileCol] = sb.toString();
            csvWriter.writeNext(nextRecord);
        }
        csvReader.close();
        csvWriter.close();
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
                    tweets.add(text);
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
