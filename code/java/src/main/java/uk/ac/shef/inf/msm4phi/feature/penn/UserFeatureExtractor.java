package uk.ac.shef.inf.msm4phi.feature.penn;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
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
        ufe.export(14, 40, userProfileCSV, tweetSolr, userSolr, outFile);
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

        String[] header = {"user", "user_name_length", "num_char_in_username", "alph_char_in_username", //0
                "username_cap", "avatar", "followers", //4
                "frieds", "friends-follower-ratio", "location", //7

                "total_tweets", "retweets", "retweet_frac", //10
                "mean_hashtag", "mean_url", "avg_time_btw_tweets", //13
                "std_time_btw_tweets", "avg_tweets_per_day", "std_tweets_per_day", "label"};//16
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

        String[] features = new String[18];
        for (int i = 0; i < features.length; i++)
            features[i] = "0";
        generateUserFeatures(u, features);
        generateTweetFeatures(tweets, features);
        return features;
    }

    private Object[] parseUsername(String username) {
        int count_alph = 0, count_digit = 0, count_uppercase = 0;
        for (int i = 0; i < username.length(); i++) {
            char c = username.charAt(i);
            if (Character.isAlphabetic(c)) {
                count_alph++;
                if (Character.isUpperCase(c))
                    count_uppercase++;
            } else if (Character.isDigit(c)) {
                count_digit++;
            }
        }
        Object[] res = new Object[3];
        res[0] = count_alph;
        res[1] = count_digit;
        if (count_uppercase == count_alph)
            res[2] = "1"; //all cap
        else if (count_uppercase == 0)
            res[2] = "2";//all lower
        else
            res[2] = "3";//mixed
        return res;
    }

    /*
    String[] header = {"user","user_name_length", "num_char_in_username", "alph_char_in_username", //0
                "username_cap", "avatar", "followers", //3
                "friends", "friends-follower-ratio", "location", //6

                "total_tweets", "retweets", "retweet_frac", //9
                "mean_hashtag", "mean_url", "avg_time_btw_tweets", //12
                "std_time_btw_tweets", "avg_tweets_per_day", "std_tweets_per_day"};//15
     */
    private void generateUserFeatures(SolrDocument u, String[] features) {
        String username = getFieldValueStr(u, "user_name");
        Object[] username_features = parseUsername(username);

        String avatar = getFieldValueStr(u, "profile_image_url");
        if (!avatar.equals("0"))
            avatar = "1";
        String location = getFieldValueStr(u, "user_location");
        if (!location.equals("0"))
            location = "1";

        String followers_count = getFieldValueStr(u, "user_followers_count");
        String friends_count = getFieldValueStr(u, "user_friends_count");

        String retweet_count = getFieldValueStr(u, "user_retweet_count");
        String statuses_count = getFieldValueStr(u, "user_newtweet_count");
        statuses_count = String.valueOf(Long.valueOf(statuses_count) + Long.valueOf(retweet_count));

        features[0] = String.valueOf(username.length());
        features[1] = String.valueOf(username_features[1]);
        features[2] = String.valueOf(username_features[0]);
        features[3] = String.valueOf(username_features[2]);
        features[4] = avatar;
        features[5] = followers_count;
        features[6] = friends_count;
        features[7] = followers_count.equalsIgnoreCase("0") ?
                "0" : String.valueOf(Double.valueOf(friends_count) / Double.valueOf(followers_count));
        features[8] = location;

        features[9] = statuses_count;
        features[10] = retweet_count;
        features[11] = statuses_count.equalsIgnoreCase("0") ? "0" :
                String.valueOf(Double.valueOf(retweet_count) / Double.valueOf(statuses_count));
    }

    /*
    "mean_hashtag", "mean_url", "avg_time_btw_tweets", //12
                "std_time_btw_tweets", "avg_tweets_per_day", "std_tweets_per_day"};//15
     */
    private void generateTweetFeatures(List<SolrDocument> tweets, String[] features) {
        double[] hashtags = new double[tweets.size()];
        double[] urls = new double[tweets.size()];

        Map<String, Integer> tweets_on_day = new HashMap<>();
        final Map<String, Long> tweets_timestamp=new HashMap<>();
        int index = 0;
        for (SolrDocument twt : tweets) {
            String id=twt.getFieldValue("id").toString();
            List<String> tags = getFieldValuesCollection(twt, "entities_hashtag");
            List<String> urls_ = getFieldValuesCollection(twt, "entities_url");
            Date d = (Date) twt.getFieldValue("created_at");
            tweets_timestamp.put(id, d.getTime());
            Calendar calendar = Calendar.getInstance();
            calendar.setTime(d);
            int dayOfMonth = calendar.get(Calendar.DAY_OF_MONTH); //169
            int month = calendar.get(Calendar.MONTH); // 5

            String day = month+"_"+dayOfMonth;
            if (tweets_on_day.containsKey(day)){
                int freq = tweets_on_day.get(day);
                freq++;
                tweets_on_day.put(day, freq);
            }else{
                tweets_on_day.put(day,1);
            }

            hashtags[index] = tags.size();
            urls[index] = urls_.size();

            index++;
        }

        double[] time_between;
        if (tweets.size()>0)
            time_between=new double[tweets.size()-1];
        else
            time_between=new double[0];
        List<String> tweetIDs=new ArrayList<>(tweets_timestamp.keySet());
        Collections.sort(tweetIDs, Comparator.comparing(tweets_timestamp::get));
        if (tweetIDs.size()>1){
            for (int i=1; i<tweetIDs.size();i++){

                long prev_time=tweets_timestamp.get(tweetIDs.get(i-1));
                long next_time=tweets_timestamp.get(tweetIDs.get(i));
                long diff=next_time-prev_time;
                time_between[i-1]=(double)diff/1000;
            }
        }

        List<Integer> tweets_of_day=new ArrayList<>(tweets_on_day.values());
        double[] array = new double[tweets_of_day.size()];
        for (int i=0; i<tweets_of_day.size(); i++){
            array[i]=(double)tweets_of_day.get(i);
        }

        Mean mean = new Mean();

        StandardDeviation std = new StandardDeviation();
        features[12]=String.valueOf(mean.evaluate(hashtags));
        features[13]=String.valueOf(mean.evaluate(urls));
        if (tweets.size()<2){
            features[14]="0";
            features[15]="0";
        }else{
            features[14]=String.valueOf(mean.evaluate(time_between));
            features[15]=String.valueOf(std.evaluate(time_between));
        }
        features[16]=String.valueOf(mean.evaluate(array));
        features[17]=String.valueOf(std.evaluate(array));
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


}
