package uk.ac.shef.inf.msm4phi.feature.uddin;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.summary.Sum;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;
import org.apache.solr.core.CoreContainer;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Extract all features except:
 * - ot/rt similarity (use UserTweetExporter to export then python kim_feature_extractor to extract
 * - features that are deprecated (see OIR paper)
 * - features that have 0 importance (see Kim et al. 2014)
 */
public class UserFeatureExtractor {

    private static final Logger LOG = Logger.getLogger(UserFeatureExtractor.class.getName());
    private Map<String, String> shortenedURLMap = new HashMap<>();

    public static void main(String[] args) throws IOException {
        UserFeatureExtractor ufe = new UserFeatureExtractor();
        CoreContainer solrContainer = new CoreContainer(args[0]);
        solrContainer.load();

        SolrClient tweetSolr = new EmbeddedSolrServer(solrContainer.getCore("tweets"));
        SolrClient userSolr = new EmbeddedSolrServer(solrContainer.getCore("users"));

        String userProfileCSV = args[1];
        String outFile = args[2];
        ufe.export(14, 40, userProfileCSV, tweetSolr, userSolr,
                outFile);
        System.exit(0);
    }

    public void export(int userCol, int labelCol, String userProfileCSV,
                       SolrClient tweetIndex, SolrClient userIndex,
                       String outFile) throws IOException {
        Reader reader = Files.newBufferedReader(Paths.get(userProfileCSV));
        CSVReader csvReader = new CSVReader(reader);

        Writer writer = Files.newBufferedWriter(Paths.get(outFile), Charset.forName("utf-8"));

        CSVWriter csvWriter = new CSVWriter(writer,
                CSVWriter.DEFAULT_SEPARATOR,
                CSVWriter.DEFAULT_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END);

        String[] header = {"user", "favorites_count", "plain_statuses", //0
                "replies_received", "replies_given", "retweets",//3
                "total_mentions", "total_urls", "total_hashtags",//6
                "tweet_influence", "std_urls", "std_hashtags",//9
                "user_activeness", "user_inclination", "collective_influence",
        "label"};//12
        // Reading Records One by One in a String array
        csvWriter.writeNext(header);
        String[] nextRecord;
        int count = 0;
        while ((nextRecord = csvReader.readNext()) != null) {
            count++;
            if (count == 1) {
                continue;
            }

            String user = nextRecord[userCol];
            String label = nextRecord[labelCol];

            String[] features = extractFeatures(user, tweetIndex, userIndex);
            String[] line = new String[features.length + 2];
            line[0] = user;
            for (int i = 0; i < features.length; i++)
                line[i + 1] = features[i];
            line[line.length - 1] = label;
            csvWriter.writeNext(line);

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
    public String[] extractFeatures(String user, SolrClient tweetIndex, SolrClient userIndex) {
        SolrQuery q = Util.createQueryTweetsOfUserScreenname(user, 5000);
        List<SolrDocument> tweets = new ArrayList<>();

        //collect all tweets
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
                    tweets.add(d);
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

        //collect user
        q = Util.createQueryByUser(user, 5000);
        QueryResponse res = null;

        SolrDocument u = null;
        try {
            res = Util.performQuery(q, userIndex);
            if (res != null)
                total = res.getResults().getNumFound();
            //update results
            if (total == 0)
                LOG.info(String.format("\t\tuser %s has no match!",
                        user));
            else {
                SolrDocumentList resultDocs = res.getResults();
                u = resultDocs.get(0);
            }
        } catch (Exception e) {
            LOG.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                    q.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
        }

        String[] features = new String[14];
        for (int i = 0; i < features.length; i++)
            features[i] = "0";
        generateUserFeatures(u, features);
        generateTweetFeatures(tweets, features);
        return features;
    }

    /*
    String[] header = {"user","favorites_count","plain_statuses", //0
                "replies_received","replies_given","retweets",//3
                "total_mentions","total_urls","total_hashtags",//6
                "tweet_influence","std_urls","std_hashtags",//9
                "user_activeness","user_inclination","collective_influence"};//12
     */
    private void generateUserFeatures(SolrDocument u, String[] features) {
        double favorites_count = getFieldValueNum(u, "user_favorited_count");
        String plain_statuses = getFieldValueStr(u, "user_newtweet_count");
        String replies_received = getFieldValueStr(u, "user_replied_count");
        String replies_given = getFieldValueStr(u, "user_reply_count");
        double retweet = getFieldValueNum(u, "user_retweet_count");
        double otweet = getFieldValueNum(u, "user_newtweet_count");
        double friends_count = getFieldValueNum(u, "user_friends_count");
        double listed_count = getFieldValueNum(u, "user_listed_count");

        double followers_count = getFieldValueNum(u, "user_followers_count");

        features[0] = String.valueOf(favorites_count);
        features[1] = plain_statuses;
        features[2] = replies_received;
        features[3] = replies_given;
        features[4] = String.valueOf(retweet);

        double collective_activeness=retweet+otweet+listed_count+friends_count;
        double inclination = 2 * retweet * otweet / (retweet + otweet);
        double collective_influence = followers_count + listed_count + favorites_count;

        features[11]=String.valueOf(collective_activeness);
        features[12] = String.valueOf(inclination);
        features[13] = String.valueOf(collective_influence);
    }


    /*
    String[] header = {
                "total_mentions","total_urls","total_hashtags",//6
                "tweet_influence","std_urls","std_hashtags",//9
                "user_activeness","user_inclination","collective_influence"};//12
     */
    private void generateTweetFeatures(List<SolrDocument> tweets, String[] features) {
        double[] hashtags = new double[tweets.size()];
        double[] urls = new double[tweets.size()];
        double[] mentions = new double[tweets.size()];

        int index = 0;
        double retweeted=0, totalRetweets=0;
        for (SolrDocument twt : tweets) {
            double retweet_count=getFieldValueNum(twt,"retweet_count");
            if (retweet_count>0) {
                retweeted++;
                totalRetweets+=retweeted;
            }

            List<String> tags = getFieldValuesCollection(twt, "entities_hashtag");
            List<String> ments = getFieldValuesCollection(twt, "entities_user_mention");
            List<String> urls_ = getFieldValuesCollection(twt, "entities_url");

            List<String> urlsExp = new ArrayList<>();
            if (urls_.size() > 0) {
                LOG.info(String.format("\t\t expanding urls %d", urls_.size()));
                for (String u : urls_) {
                    if (shortenedURLMap.containsKey(u))
                        urlsExp.add(shortenedURLMap.get(u));
                    else {
                        try {
                            String expUrl = expandUrl(u);
                            shortenedURLMap.put(u, expUrl);
                            urlsExp.add(expUrl);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }

                }
            }
            urls_ = urlsExp;

            hashtags[index] = tags.size();
            mentions[index] = ments.size();
            urls[index] = urls_.size();

            index++;
        }

        Sum sum = new Sum();
        StandardDeviation std = new StandardDeviation();

        double tweet_influence=retweeted==0?0:totalRetweets/retweeted;
        features[5] = String.valueOf(sum.evaluate(mentions));
        features[6] = String.valueOf(sum.evaluate(urls));
        features[7] = String.valueOf(sum.evaluate(hashtags));
        features[8] = String.valueOf(tweet_influence);
        features[9] = String.valueOf(std.evaluate(urls));
        features[10] = String.valueOf(std.evaluate(hashtags));
    }

    public static String expandUrl(String shortenedUrl) throws IOException {
//        URL url = new URL(shortenedUrl);
//        // open connection
//        HttpURLConnection httpURLConnection = (HttpURLConnection) url.openConnection(Proxy.NO_PROXY);
//
//        // stop following browser redirect
//        httpURLConnection.setInstanceFollowRedirects(false);
//
//        // extract location header containing the actual destination URL
//        String expandedURL = httpURLConnection.getHeaderField("Location");
//        httpURLConnection.disconnect();
//
//        return expandedURL;
        return shortenedUrl;
    }


    private String getFieldValueStr(SolrDocument doc, String field) {
        Object v = doc.getFieldValue(field);
        if (v == null)
            return "0";
        else
            return v.toString();
    }

    private List<String> getFieldValuesCollection(SolrDocument doc, String field) {
        Collection<Object> values = doc.getFieldValues(field);
        if (values == null)
            return new ArrayList<>();
        List<String> strVals = new ArrayList<>();
        for (Object o : values)
            strVals.add(o.toString());
        return strVals;
    }

    private double getFieldValueNum(SolrDocument doc, String field) {
        Object v = doc.getFieldValue(field);
        if (v == null)
            return 0;
        else
            return Double.valueOf(v.toString());
    }
}

