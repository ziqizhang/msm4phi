package uk.ac.shef.inf.msm4phi.analysis;

import com.opencsv.CSVWriter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.IOException;
import java.util.*;

public class UserFeatureExporter {

    private static final int resultBatchSize = 10000;
    private static final Logger LOG = Logger.getLogger(UserFeatureExporter.class.getName());

    public void process(int sampleSize, String outFile,
                        SolrClient userCore) throws IOException, SolrServerException {
        LOG.info("Counting users ...");
        List<String> userIDs = new ArrayList<>();
        SolrQuery query = new SolrQuery();
        query.setQuery("*:*");
        query.setStart(0);
        query.setRows(resultBatchSize);
        long users = 0;
        boolean stop = false;
        while (!stop) {
            QueryResponse qr = null;
            try {
                qr = Util.performQuery(query, userCore);
                if (qr != null)
                    users = qr.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        users, query.getStart(), query.getStart() + query.getRows()));
                for (SolrDocument d : qr.getResults()) {
                    Object username = d.getFieldValue("id");
                    Object newTweets = d.getFieldValue("user_newtweet_count");
                    Object retweets =d.getFieldValue("user_retweet_count");

                    int total=0;
                    if (newTweets!=null)
                        total+=Integer.valueOf(newTweets.toString());
                    if (retweets!=null)
                        total+=Integer.valueOf(retweets.toString());

                    if (total>9)
                        userIDs.add(username.toString());
                }
            } catch (Exception e) {
                LOG.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                        query.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
            }

            int curr = query.getStart() + query.getRows();
            if (curr < users)
                query.setStart(curr);
            else
                stop = true;
        }


        Collections.shuffle(userIDs);

        String[] header = new String[27];
        header[0] = "twitter_id";
        header[1] = "user_screen_name";
        header[2] = "user_name";
        header[3] = "user_statuses_count";
        header[4] = "user_friends_count";
        header[5] = "user_followers_count";
        header[6] = "user_listed_count";
        header[7] = "user_location";
        header[8] = "user_favorites_count";
        header[9] = "user_desc";
        header[10] = "user_url";
        header[11] = "profile_background_image_url";
        header[12] = "profile_image_url";
        header[13] = "user_newtweet_count";
        header[14] = "user_retweet_count";
        header[15] = "user_reply_count";
        header[16] = "user_quote_count";
        header[17] = "user_favorited_count";
        header[18] = "user_retweeted_count";
        header[19] = "user_quoted_count";
        header[20] = "user_replied_count";
        header[21] = "user_entities_hashtag";
        header[22] = "user_entities_symbol";
        header[23] = "user_entities_url";
        header[24] = "user_entities_user_mention";
        header[25] = "user_entities_media_url";
        header[26] = "user_entities_media_type";

        CSVWriter csvWriter = Util.createCSVWriter(outFile);
        csvWriter.writeNext(header);
        for (int i = 0; i < sampleSize && i < userIDs.size(); i++) {
            String targetID = userIDs.get(i);
            query = new SolrQuery();
            query.setQuery("id:" + targetID);
            query.setStart(0);
            query.setRows(1);
            QueryResponse qr = Util.performQuery(query, userCore);
            SolrDocument doc = qr.getResults().get(0);
            String[] values = new String[27];
            String username = getValue(doc.getFieldValue("user_screen_name"));
            if (username.equalsIgnoreCase("null"))
                continue;
            values[0] = "https://twitter.com/" +username;
            values[1] = getValue(doc.getFieldValue("user_screen_name"));
            values[2] = getValue(doc.getFieldValue("user_name"));
            values[3] = getValue(doc.getFieldValue("user_statuses_count"));
            values[4] = getValue(doc.getFieldValue("user_friends_count"));
            values[5] = getValue(doc.getFieldValue("user_followers_count"));
            values[6] = getValue(doc.getFieldValue("user_listed_count"));
            values[7] = getValue(doc.getFieldValue("user_location"));
            values[8] = getValue(doc.getFieldValue("user_favorites_count"));
            values[9] = getValue(doc.getFieldValue("user_desc"));
            values[10] = getValue(doc.getFieldValue("user_url"));
            values[11] = getValue(doc.getFieldValue("profile_background_image_url"));
            values[12] = getValue(doc.getFieldValue("profile_image_url"));
            values[13] = getValue(doc.getFieldValue("user_newtweet_count"));
            values[14] = getValue(doc.getFieldValue("user_retweet_count"));
            values[15] = getValue(doc.getFieldValue("user_reply_count"));
            values[16] = getValue(doc.getFieldValue("user_quote_count"));
            values[17] = getValue(doc.getFieldValue("user_favorited_count"));
            values[18] = getValue(doc.getFieldValue("user_retweeted_count"));
            values[19] = getValue(doc.getFieldValue("user_quoted_count"));
            values[20] = getValue(doc.getFieldValue("user_replied_count"));
            values[21] = countValues(doc.getFieldValues("user_entities_hashtag"));
            values[22] = countValues(doc.getFieldValues("user_entities_symbol"));
            values[23] = countValues(doc.getFieldValues("user_entities_url"));
            values[24] = countValues(doc.getFieldValues("user_entities_user_mention"));
            values[25] = countValues(doc.getFieldValues("user_entities_media_url"));
            values[26] = getValue(doc.getFieldValue("user_entities_media_type"));
            csvWriter.writeNext(values);

        }
        csvWriter.close();

    }


    private String getValue(Object o) {
        if (o == null)
            return "NULL";
        return o.toString().replaceAll("\\s+", " ").trim();
    }

    private String countValues(Collection o) {
        if (o == null)
            return "0";
        return String.valueOf(o.size());
    }
}
