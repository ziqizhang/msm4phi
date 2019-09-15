package uk.ac.shef.inf.msm4phi.feature.kim;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.apache.commons.math3.stat.descriptive.rank.Min;
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

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Extract all features except:
 * - ot/rt similarity (use UserTweetExporter to export then python kim_feature_extractor to extract
 * - features that are deprecated (see OIR paper)
 * - features that have 0 importance (see Kim et al. 2014)
 */
public class UserFeatureExtractor {


    private static final Logger LOG = Logger.getLogger(UserFeatureExtractor.class.getName());
    private Map<String, String> shortenedURLMap=new HashMap<>();

    public static void main(String[] args) throws IOException {
        UserFeatureExtractor ufe = new UserFeatureExtractor();
        CoreContainer solrContainer = new CoreContainer(args[0]);
        solrContainer.load();

        SolrClient tweetSolr = new EmbeddedSolrServer(solrContainer.getCore("tweets"));
        SolrClient userSolr = new EmbeddedSolrServer(solrContainer.getCore("users"));

        List<String> lines=FileUtils.readLines(new File(args[1]));
        Set<String> keywords=new HashSet<String>();
        for(String l : lines)
            keywords.add(l.split(",")[0].trim().toLowerCase());
        String userProfileCSV=args[2];
        String outFile=args[3];
        ufe.export(14, 40,userProfileCSV, tweetSolr, userSolr,
                new ArrayList<>(keywords), outFile);
        System.exit(0);
    }

    public void export(int userCol, int labelCol,String userProfileCSV,
                       SolrClient tweetIndex, SolrClient userIndex,
                       List<String> keywords,
                       String outFile) throws IOException {
        Reader reader = Files.newBufferedReader(Paths.get(userProfileCSV));
        CSVReader csvReader = new CSVReader(reader);

        Writer writer = Files.newBufferedWriter(Paths.get(outFile), Charset.forName("utf-8"));

        CSVWriter csvWriter = new CSVWriter(writer,
                CSVWriter.DEFAULT_SEPARATOR,
                CSVWriter.DEFAULT_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END);

        String[] header = {"user","default_profile_image", "favorite_count", "favorites_count", //0
                "followers_count", "friends_count", "listed_count", //3
                "profile_use_background_image", "retweet_count", "statuses_count", //6

                "ot_count", "ot_favorite_count", "total_hashtag_in_ot", //9
                "total_url_in_ot", "total_mention_in_ot", "mean_hashtag_in_ot", //12
                "mean_url_in_ot", "mean_mention_in_ot", "median_hashtag_in_ot", //15
                "median_url_in_ot", "median_mention_in_ot", "max_hashtag_in_ot",//18
                "max_url_in_ot", "max_mention_in_ot", "min_hashtag_in_ot", //21
                "min_url_in_ot", "min_mention_in_ot", "total_keyword_in_ot", //24
                "unique_keyword_in_ot", "total_keyword_in_hashtag_in_ot", "unique_keyword_in_hashtag_in_ot",//27
                "total_keyword_in_url_in_ot", "unique_keyword_in_url_in_ot", "total_keyword_in_mention_in_ot",//30
                "unique_keyword_in_mention_in_ot",//33

                "rt_count", "rt_favorite_count", "total_hashtag_in_rt", //34
                "total_url_in_rt", "total_mention_in_rt", "mean_hashtag_in_rt", //37
                "mean_url_in_rt", "mean_mention_in_rt", "median_hashtag_in_rt",//40
                "median_url_in_rt", "median_mention_in_rt", "max_hashtag_in_rt",//43
                "max_url_in_rt", "max_mention_in_rt", "min_hashtag_in_rt",//46
                "min_url_in_rt", "min_mention_in_rt", "total_keyword_in_rt",//49
                "unique_keyword_in_rt", "total_keyword_in_hashtag_in_rt", "unique_keyword_in_hashtag_in_rt",//52
                "total_keyword_in_url_in_rt", "unique_keyword_in_url_in_rt", "total_keyword_in_mention_in_rt",//55
                "unique_keyword_in_mention_in_rt",//58

                "screen_name_keyword", "description_keyword","label"};//59
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
            String label=nextRecord[labelCol];

            String[] features = extractFeatures(user, tweetIndex, userIndex, keywords);
            String[] line=new String[features.length+2];
            line[0]=user;
            for (int i=0;i<features.length;i++)
                line[i+1]=features[i];
            line[line.length-1]=label;
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
    public String[] extractFeatures(String user, SolrClient tweetIndex, SolrClient userIndex, List<String> keywords) {
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

        String[] features = new String[61];
        for (int i = 0; i < features.length; i++)
            features[i] = "0";
        generateUserFeatures(u, features, keywords);
        generateTweetFeatures(tweets, features, keywords);
        return features;
    }

    private void generateUserFeatures(SolrDocument u, String[] features, List<String> keywords) {
        String default_profile_image = getFieldValueStr(u, "profile_image_url");
        if (!default_profile_image.equals("0"))
            default_profile_image = "1";
        String favorite_count = getFieldValueStr(u, "user_favorited_count");
        String favorites_count = getFieldValueStr(u, "user_favorites_count");
        String followers_count = getFieldValueStr(u, "user_followers_count");
        String friends_count = getFieldValueStr(u, "user_friends_count");
        String listed_count = getFieldValueStr(u, "user_listed_count");
        String profile_use_bg_image = getFieldValueStr(u, "profile_background_image_url");
        if (!profile_use_bg_image.equals("0"))
            profile_use_bg_image = "1";
        String retweet_count = getFieldValueStr(u, "user_retweet_count");
        String statuses_count = getFieldValueStr(u, "user_newtweet_count");
        statuses_count = String.valueOf(Long.valueOf(statuses_count) + Long.valueOf(retweet_count));

        features[0] = default_profile_image;
        features[1] = favorite_count;
        features[2] = favorites_count;
        features[3] = followers_count;
        features[4] = friends_count;
        features[5] = listed_count;
        features[6] = profile_use_bg_image;
        features[7] = retweet_count;
        features[8] = statuses_count;

        String screenname=getFieldValueStr(u,"user_screen_name");
        features[59]=String.valueOf(screenname.split("\\s+").length);
        String desc=getFieldValueStr(u,"user_desc");
        features[60]=String.valueOf(desc.split("\\s+").length);
    }

    private void generateTweetFeatures(List<SolrDocument> tweets, String[] features, List<String> keywords) {
        double[] ot_hashtags=new double[tweets.size()];
        double[] ot_urls=new double[tweets.size()];
        double[] ot_mentions=new double[tweets.size()];
        StringBuilder ot_str=new StringBuilder();
        StringBuilder ot_hashtagStr=new StringBuilder();
        StringBuilder ot_urlStr=new StringBuilder();
        StringBuilder ot_mentionStr=new StringBuilder();

        double[] rt_hashtags=new double[tweets.size()];
        double[] rt_urls=new double[tweets.size()];
        double[] rt_mentions=new double[tweets.size()];
        StringBuilder rt_str=new StringBuilder();
        StringBuilder rt_hashtagStr=new StringBuilder();
        StringBuilder rt_urlStr=new StringBuilder();
        StringBuilder rt_mentionStr=new StringBuilder();

        double ot_count = 0, ot_favorite_count = 0, total_hashtag_in_ot = 0, //9
                total_url_in_ot = 0, total_mention_in_ot = 0, mean_hashtag_in_ot = 0, //12
                mean_url_in_ot = 0, mean_mention_in_ot = 0, median_hashtag_in_ot = 0, //15
                median_url_in_ot = 0, median_mention_in_ot = 0, max_hashtag_in_ot = 0,//18
                max_url_in_ot = 0, max_mention_in_ot = 0, min_hashtag_in_ot = 0, //21
                min_url_in_ot = 0, min_mention_in_ot = 0;

        double rt_count = 0, rt_favorite_count = 0, trtal_hashtag_in_rt = 0, //9
                trtal_url_in_rt = 0, trtal_mention_in_rt = 0, mean_hashtag_in_rt = 0, //12
                mean_url_in_rt = 0, mean_mention_in_rt = 0, median_hashtag_in_rt = 0, //15
                median_url_in_rt = 0, median_mention_in_rt = 0, max_hashtag_in_rt = 0,//18
                max_url_in_rt = 0, max_mention_in_rt = 0, min_hashtag_in_rt = 0, //21
                min_url_in_rt = 0, min_mention_in_rt = 0;


        int index=0;
        for (SolrDocument twt: tweets){
            boolean isOT= getFieldValueStr(twt, "retweeted_status_id_str").equals("0");
            double favorite_count=getFieldValueNum(twt,"favorite_count");

            if (favorite_count>1.0)
                System.out.println("ok");

            List<String> tags=getFieldValuesCollection(twt,"entities_hashtag");
            List<String> ments=getFieldValuesCollection(twt,"entities_user_mention");
            List<String> urls_=getFieldValuesCollection(twt,"entities_url");

            List<String> urlsExp=new ArrayList<>();
            if (urls_.size()>0){
                LOG.info(String.format("\t\t expanding urls %d", urls_.size()));
                for(String u : urls_){
                    if (shortenedURLMap.containsKey(u))
                        urlsExp.add(shortenedURLMap.get(u));
                    else{
                        try {
                            String expUrl=expandUrl(u);
                            shortenedURLMap.put(u, expUrl);
                            urlsExp.add(expUrl);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }

                }
            }
            urls_=urlsExp;

            if (isOT){
                ot_count++;
                ot_favorite_count+=favorite_count;
                ot_hashtags[index]=tags.size();
                ot_mentions[index]=ments.size();
                ot_urls[index]=urls_.size();
                addToString(tags, ot_hashtagStr);
                addToString(ments, ot_mentionStr);
                addToString(urls_, ot_urlStr);
                ot_str.append(getFieldValueStr(twt,"status_text")).append(" ");
            }
            else{
                rt_count++;
                rt_favorite_count+=favorite_count;
                rt_hashtags[index]=tags.size();
                rt_mentions[index]=ments.size();
                rt_urls[index]=urls_.size();
                addToString(tags, rt_hashtagStr);
                addToString(ments, rt_mentionStr);
                addToString(urls_, rt_urlStr);
                rt_str.append(getFieldValueStr(twt,"status_text")).append(" ");
            }
            index++;
        }

        Median median = new Median();
        Mean mean = new Mean();
        Max max = new Max();
        Min min = new Min();
        Sum sum = new Sum();

        total_hashtag_in_ot = sum.evaluate(ot_hashtags);
        total_url_in_ot=sum.evaluate(ot_urls);
        total_mention_in_ot=sum.evaluate(ot_urls);
        mean_hashtag_in_ot=mean.evaluate(ot_hashtags);
        mean_url_in_ot=mean.evaluate(ot_urls);
        mean_mention_in_ot=mean.evaluate(ot_mentions);
        median_hashtag_in_ot=median.evaluate(ot_hashtags);
        median_mention_in_ot=median.evaluate(ot_mentions);
        median_url_in_ot=median.evaluate(ot_urls);
        max_hashtag_in_ot=max.evaluate(ot_hashtags);
        max_url_in_ot=max.evaluate(ot_urls);
        max_mention_in_ot=max.evaluate(ot_mentions);
        min_hashtag_in_ot=min.evaluate(ot_hashtags);
        min_url_in_ot=min.evaluate(ot_urls);
        min_mention_in_ot=min.evaluate(ot_mentions);

        trtal_hashtag_in_rt = sum.evaluate(rt_hashtags);
        trtal_url_in_rt=sum.evaluate(rt_urls);
        trtal_mention_in_rt=sum.evaluate(rt_urls);
        mean_hashtag_in_rt=mean.evaluate(rt_hashtags);
        mean_url_in_rt=mean.evaluate(rt_urls);
        mean_mention_in_rt=mean.evaluate(rt_mentions);
        median_hashtag_in_rt=median.evaluate(rt_hashtags);
        median_mention_in_rt=median.evaluate(rt_mentions);
        median_url_in_rt=median.evaluate(rt_urls);
        max_hashtag_in_rt=max.evaluate(rt_hashtags);
        max_url_in_rt=max.evaluate(rt_urls);
        max_mention_in_rt=max.evaluate(rt_mentions);
        min_hashtag_in_rt=min.evaluate(rt_hashtags);
        min_url_in_rt=min.evaluate(rt_urls);
        min_mention_in_rt=min.evaluate(rt_mentions);

        List<String> ot_hashtag_match=findMatchingKeywords(keywords, ot_hashtagStr.toString());
        List<String> rt_hashtag_match=findMatchingKeywords(keywords, rt_hashtagStr.toString());
        List<String> ot_mention_match=findMatchingKeywords(keywords, ot_mentionStr.toString());
        List<String> rt_mention_match=findMatchingKeywords(keywords, rt_mentionStr.toString());
        List<String> ot_url_match=findMatchingKeywords(keywords, ot_urlStr.toString());
        List<String> rt_url_match=findMatchingKeywords(keywords, rt_urlStr.toString());
        List<String> ot_match=findMatchingKeywords(keywords, ot_str.toString());
        List<String> rt_match=findMatchingKeywords(keywords, rt_str.toString());

        features[9]=String.valueOf(ot_count);
        features[10]=String.valueOf(ot_favorite_count);
        features[11]=String.valueOf(total_hashtag_in_ot);
        features[12]=String.valueOf(total_url_in_ot);
        features[13]=String.valueOf(total_mention_in_ot);
        features[14]=String.valueOf(mean_hashtag_in_ot);
        features[15]=String.valueOf(mean_url_in_ot);
        features[16]=String.valueOf(mean_mention_in_ot);
        features[17]=String.valueOf(median_hashtag_in_ot);
        features[18]=String.valueOf(median_url_in_ot);
        features[19]=String.valueOf(median_mention_in_ot);
        features[20]=String.valueOf(max_hashtag_in_ot);
        features[21]=String.valueOf(max_url_in_ot);
        features[22]=String.valueOf(max_mention_in_ot);
        features[23]=String.valueOf(min_hashtag_in_ot);
        features[24]=String.valueOf(min_url_in_ot);
        features[25]=String.valueOf(min_mention_in_ot);
        features[26]=String.valueOf(ot_match.size());
        features[27]=String.valueOf(new HashSet(ot_match).size());
        features[28]=String.valueOf(ot_hashtag_match.size());
        features[29]=String.valueOf(new HashSet(ot_hashtag_match).size());
        features[30]=String.valueOf(ot_url_match.size());
        features[31]=String.valueOf(new HashSet(ot_url_match).size());
        features[32]=String.valueOf(ot_mention_match.size());
        features[33]=String.valueOf(new HashSet(ot_mention_match).size());
        features[34]=String.valueOf(rt_count);
        features[35]=String.valueOf(rt_favorite_count);
        features[36]=String.valueOf(trtal_hashtag_in_rt);
        features[37]=String.valueOf(trtal_url_in_rt);
        features[38]=String.valueOf(trtal_mention_in_rt);
        features[39]=String.valueOf(mean_hashtag_in_rt);
        features[40]=String.valueOf(mean_url_in_rt);
        features[41]=String.valueOf(mean_mention_in_rt);
        features[42]=String.valueOf(median_hashtag_in_rt);
        features[43]=String.valueOf(median_url_in_rt);
        features[44]=String.valueOf(median_mention_in_rt);
        features[45]=String.valueOf(max_hashtag_in_rt);
        features[46]=String.valueOf(max_url_in_rt);
        features[47]=String.valueOf(max_mention_in_rt);
        features[48]=String.valueOf(min_hashtag_in_rt);
        features[49]=String.valueOf(min_url_in_rt);
        features[50]=String.valueOf(min_mention_in_rt);
        features[51]=String.valueOf(rt_match.size());
        features[52]=String.valueOf(new HashSet(rt_match).size());
        features[53]=String.valueOf(rt_hashtag_match.size());
        features[54]=String.valueOf(new HashSet(rt_hashtag_match).size());
        features[55]=String.valueOf(rt_url_match.size());
        features[56]=String.valueOf(new HashSet(rt_url_match).size());
        features[57]=String.valueOf(rt_mention_match.size());
        features[58]=String.valueOf(new HashSet(rt_mention_match).size());
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

    private List<String> findMatchingKeywords(List<String> keywords, String text){
        text=text.toLowerCase();
        List<String> res = new ArrayList<>();
        for(String k : keywords) {
            Pattern p = Pattern.compile(k);
            Matcher m=p.matcher(text);
            while(m.find()){
                int s=m.start();
                int e=m.end();
                String word=text.substring(s,e);
                res.add(word);
            }
        }

        return res;
    }

    private void addToString(List<String> values, StringBuilder sb){
        for(String v: values)
            sb.append(v).append(" ");
    }

    private String getFieldValueStr(SolrDocument doc, String field) {
        Object v = doc.getFieldValue(field);
        if (v == null)
            return "0";
        else
            return v.toString();
    }

    private List<String> getFieldValuesCollection(SolrDocument doc, String field){
        Collection<Object> values= doc.getFieldValues(field);
        if (values==null)
            return new ArrayList<>();
        List<String> strVals=new ArrayList<>();
        for (Object o: values)
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

