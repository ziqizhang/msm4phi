package uk.ac.shef.inf.msm4phi;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.opencsv.*;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import twitter4j.Twitter;
import twitter4j.TwitterFactory;
import twitter4j.conf.ConfigurationBuilder;

public class Util {

    public static Twitter authenticateTwitter(String consumerKey,
                                           String consumerSecret,
                                           String accessKey,
                                           String accessSecrete){
        ConfigurationBuilder cb = new ConfigurationBuilder();
        cb.setDebugEnabled(true)
                .setOAuthConsumerKey(consumerKey)
                .setOAuthConsumerSecret(consumerSecret)
                .setOAuthAccessToken(accessKey)
                .setOAuthAccessTokenSecret(accessSecrete);
        TwitterFactory tf = new TwitterFactory(cb.build());
        return tf.getInstance();
    }

    public static URL expandShortenedURL(String strURL) throws IOException {

        URLConnection conn = null;

        URL inputURL = new URL(strURL);
        conn = inputURL.openConnection();
        return conn.getURL();
    }

    public static SolrClient getSolrClient(Path solrHome, String coreName) {
        SolrClient solrClient = new EmbeddedSolrServer(solrHome, coreName);
        return solrClient;
    }

    public static SolrQuery createQueryTweetsOfHashtags(int resultBatchSize, String... hashtags) {
        SolrQuery query = new SolrQuery();
        StringBuilder qValues = new StringBuilder();

        int count = 0;
        for (String h : hashtags) {
            count++;
            if (count == 1)
                qValues.append(h);
            else
                qValues.append(" OR ").append(h);
        }
        query.setQuery("entities_hashtag:(" + qValues.toString() + ")");
        query.setStart(0);
        query.setRows(resultBatchSize);
        return query;
    }

    public static SolrQuery createQueryTweetsOfUser(int resultBatchSize, String userID) {
        SolrQuery query = new SolrQuery();
        query.setQuery("user_id_str:"+userID);
        query.setStart(0);
        query.setRows(resultBatchSize);
        return query;
    }

    public static QueryResponse performQuery(SolrQuery q, SolrClient solrClient) throws IOException, SolrServerException {
        return solrClient.query(q);
    }

    /**
     *
     * @param hashtagFile should have the same format as in '2_PART2_processed_hashtags.tsv'
     * @return
     */
    public static Map<String, List<String>> readHashtags(File hashtagFile) throws IOException {
        Map<String, List<String>> result = new HashMap<>();
        Reader reader = Files.newBufferedReader(Paths.get(hashtagFile.getPath()));
        CSVParser parser = new CSVParserBuilder()
                .withSeparator('\t').build();
        CSVReader csvReader =
                new CSVReaderBuilder(reader)
                        .withSkipLines(1)
                        .withCSVParser(parser)
                        .build();
        String[] nextRecord;
        while ((nextRecord = csvReader.readNext()) != null) {
            String k = nextRecord[0];
            List<String> values=new ArrayList<>();
            for (String tag : nextRecord[1].split("\\|")){
                if (tag.length()>0)
                    values.add(tag.substring(1).toLowerCase());
            }

            if (nextRecord.length>1){
                for (String tag : nextRecord[2].split("\\|")){
                    if (tag.length()>0)
                        values.add(tag.substring(1).toLowerCase());
                }
            }
            result.put(k, values);
        }
        return result;
    }

    public static CSVWriter createCSVWriter(String fullFilePath) throws IOException {
        Writer writer = Files.newBufferedWriter(Paths.get(fullFilePath));

        CSVWriter csvWriter = new CSVWriter(writer,
                CSVWriter.DEFAULT_SEPARATOR,
                CSVWriter.DEFAULT_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END);
        return csvWriter;
    }

}
