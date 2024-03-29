package uk.ac.shef.inf.msm4phi.export;

import com.opencsv.CSVWriter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;
import uk.ac.shef.inf.msm4phi.IndexAnalyserWorker;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.IOException;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.*;

public class WorkerTweetExportNodeXL extends IndexAnalyserWorker {
    final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("dd-MM-yyyy HH:mm:ss");
    private static final Logger LOG = Logger.getLogger(WorkerTweetExportNodeXL.class.getName());
    private int count=1;
    private int currLine=0;
    private int maxLines=10000;
    public WorkerTweetExportNodeXL(int id, SolrClient solrClient, String outFolder) {
        super(id, solrClient, outFolder);
    }


    public int getResultBatchSize() {
        return this.resultBatchSize;
    }

    public void setResultBatchSize(int resultBatchSize) {
        this.resultBatchSize = resultBatchSize;
    }

    /**
     * Query the solr backend to process tweets
     *
     * @param tasks
     * @return
     */
    protected int computeSingleWorker(Map<String, List<String>> tasks) {
        for (Map.Entry<String, List<String>> en : tasks.entrySet()) {
            count=0;
            currLine=0;
            LOG.info(String.format("\t processing hashtag '%s' with %d variants...", en.getKey(), en.getValue().size()));
            SolrQuery q = Util.createQueryTweetsOfHashtags(resultBatchSize, en.getValue().toArray(new String[0]));

            CSVWriter csvWriter = null;
            String tag=en.getKey();
            if (tag.contains("appendicitis"))
                System.out.println();
            try {
                csvWriter = Util.createCSVWriter(outFolder + "/" + tag.
                        replaceAll("#", "_")+"_"+count+".csv");
                writeCSVHeader(csvWriter);
            } catch (IOException e) {
                LOG.warn(String.format("\tfailed due to unable to create output file at %s \n\tcontinue to the next hashtag",
                        outFolder + "/" + en.getKey().replaceAll("#", "_")));
                continue;
            }

            boolean stop = false;
            while (!stop) {
                QueryResponse res = null;
                long total = 0;
                try {
                    res = Util.performQuery(q, solrClient);
                    if (res != null)
                        total = res.getResults().getNumFound();
                    //update results
                    LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                            total, q.getStart(), q.getStart() + q.getRows()));
                    csvWriter=writeCSVContent(csvWriter, tag, res.getResults());

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
            try {
                csvWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return 0;
    }


    protected WorkerTweetExportNodeXL createInstance(Map<String, List<String>> splitTasks, int id) {
        WorkerTweetExportNodeXL worker = new WorkerTweetExportNodeXL(id, this.solrClient, outFolder);
        worker.setHashtagMap(splitTasks);
        worker.setMaxTasksPerThread(this.maxTasksPerThread);
        return worker;
    }


    /**
     * Schema:
     * <br/> vertex 1 ->user_screen_name
     * <br/> vertex 2 ->entities_user_mention or in_reply_to_screen_name
     * <br/> reciprocated -> NULL
     * <br/> NULL -> NULL
     * <br/> relationship -> Mentions or Tweet (only if no mention or reply) or 'Replies to'
     * <br/> relationship date (utc) -> created_at
     * <br/> tweet -> status_text
     * <br/> URLs in Tweet -> entities_url, separate with space
     * <br/> Domains in Tweet -> entities_url, separate with space
     * <br/> Hashtags in Tweet -> entities_hashtag, separate with space, remove #
     * <br/> Media in Tweet -> entities_media_url
     * <br/> Tweet Image File -> ?
     * <br/> Tweet Date (UTC) -> created_at
     * <br/> Twitter Page for Tweet -> generate this based on domain, user, and tweet id
     * <br/> Latitude -> coordinate_lat
     * <br/> Longitude -> coordinate_lon
     * <br/> Imported ID ->?
     * <br/> In-Reply-To-Tweet ID -> in_reply_to_status_id_str
     * <br/> Favorited -> all 0
     * <br/> Favorite Count -> favorite_count
     * <br/> In-Reply-To-User ID ->in_reply_to_user_id_str
     * <br/> Is Quote Status -> 0 or 1
     * <br/> Language -> en
     * <br/> Possibly Sensitive -> NULL
     * <br/> Quoted Status ID ->quoted_status_id_str
     * <br/> Retweeted -> all 0
     * <br/> Retweet Count -> retweet_count
     * <br/> Retweet ID -> retweeted_status_id_str
     * <br/> Source -> NULL
     * <br/> Truncated -> NULL
     * <br/> Unified Twitter ID ->id
     * <br/> Imported Tweet Type -> NULL
     *
     * @param csvWriter
     */
    private void writeCSVHeader(CSVWriter csvWriter) {
        String[] headerRecord = {
                //0         1           2                3      4               5
                "Vertex 1", "Vertex 2", "Reciprocated?", "", "Relationship", "Relationship Date (UTC)",
                //6         7              8                  9                 10
                "Tweet", "URLs in Tweet", "Domains in Tweet", "Hashtags in Tweet", "Media in Tweet",
                //11                    12              13                      14          15
                "Tweet Image File", "Tweet Date (UTC)", "Twiter Page for Tweet", "Latitude", "Longitude",
                //16            17                      18          19              20
                "Imported ID", "In-Reply-To Tweet ID", "Favorited", "Favorite Count", "In-Reply-To User ID",
                //21                22          23                  24                  25
                "Is Quote Status", "Language", "Possibly Sensitive", "Quoted Status ID", "Retweeted",
                //26                27          28      29          30
                "Retweet Count", "Retweet ID", "Source", "Truncated", "Unified Twitter ID",
                //31
                "Imported Tweet Type"};
        csvWriter.writeNext(headerRecord);
    }

    /**
     * Expected schema see method 'writeCSVHeader'
     *
     * @param csvWriter
     * @param results
     */
    private CSVWriter writeCSVContent(CSVWriter csvWriter, String tag, SolrDocumentList results) {
        for (SolrDocument d : results) {
            if (currLine>=maxLines){
                try{
                    csvWriter.close();
                    currLine=0;
                    count++;
                    csvWriter = Util.createCSVWriter(outFolder + "/" + tag.
                            replaceAll("#", "_")+"_"+count+".csv");
                    writeCSVHeader(csvWriter);
                }catch (IOException ioe){
                    ioe.printStackTrace();
                }
            }
            String vertex1 = d.getFieldValue("user_screen_name").toString();
            String reciprocated = "";
            String col3 = "";
            String relationship = "";
            Date relationshipDate = (Date) d.getFieldValue("created_at");
            String dateStr=DATE_FORMAT.format(relationshipDate);
            String tweet = d.getFieldValue("status_text").toString();
            String[] urls_and_domains = getEntityURLsAndDomains(d);
            String hashtags = getEntityHashtags(d);
            String media=getEntityMedia(d);
            String tweetImg="";
            String twitterPage=getTwitterPageForTweet(d);
            String lat = d.getFieldValue("coordinate_lat")==null?"":d.getFieldValue("coordinate_lat").toString();
            String lon = d.getFieldValue("coordinate_lon")==null?"":d.getFieldValue("coordinate_lon").toString();
            String importedID="ID_"+d.getFieldValue("id").toString();
            String inReplyToTweetID=d.getFirstValue("in_reply_to_status_id_str")==null?"":
                    "ID_"+d.getFirstValue("in_reply_to_status_id_str").toString();
            String favorited="0";
            String favoriteCount=d.getFirstValue("favorite_count")==null?"":
                    d.getFirstValue("favorite_count").toString();
            String inReplyToUserID=d.getFirstValue("in_reply_to_user_id_str")==null?"":
                    "ID_"+d.getFirstValue("in_reply_to_user_id_str").toString();
            String isQuoteStatus=d.getFirstValue("is_quote_status")==null?"":
                    d.getFirstValue("is_quote_status").toString();
            isQuoteStatus=isQuoteStatus.equalsIgnoreCase("false")?"0":"1";

            String language="en";
            String possiblySensitive="";
            String quotedStatusID=d.getFirstValue("quoted_status_id_str")==null?"":
                    "ID_"+d.getFirstValue("quoted_status_id_str").toString();
            String retweeted="0";
            String retweetedCount=d.getFirstValue("retweet_count")==null?"":
                    d.getFirstValue("retweet_count").toString();
            String retweetID=d.getFirstValue("retweeted_status_id_str")==null?"":
                    "ID_"+d.getFirstValue("retweeted_status_id_str").toString();
            String source="", truncated="", importedTweetType="";


            Collection<Object> userMentions = d.getFieldValues("entities_user_mention");
            if (userMentions != null) {
                relationship = "Mentions";
                for(Object vertex2 : userMentions) {
                    csvWriter.writeNext(new String[]{
                            vertex1, vertex2.toString(), reciprocated,col3,relationship, dateStr,
                            tweet, urls_and_domains[0], urls_and_domains[1],hashtags, media,
                            tweetImg, dateStr, twitterPage,lat, lon,importedID,
                            inReplyToTweetID, favorited,favoriteCount, inReplyToUserID,isQuoteStatus,
                            language, possiblySensitive,quotedStatusID,retweeted,
                            retweetedCount, retweetID,source,truncated,importedID,
                            importedTweetType
                    });
                    currLine++;
                }
            }
            Collection<Object> reply2UserIDs = d.getFieldValues("in_reply_to_user_id_str");
            if (reply2UserIDs != null) {
                relationship = "Replies to";
                for(Object vertex2 : reply2UserIDs) {
                    csvWriter.writeNext(new String[]{
                            vertex1, vertex2.toString(), reciprocated,col3,relationship, dateStr,
                            tweet, urls_and_domains[0], urls_and_domains[1],hashtags, media,
                            tweetImg, dateStr,twitterPage,lat, lon,importedID,
                            inReplyToTweetID, favorited,favoriteCount, inReplyToUserID,isQuoteStatus,
                            language, possiblySensitive,quotedStatusID,retweeted,
                            retweetedCount, retweetID,source,truncated,importedID,
                            importedTweetType
                    });
                    currLine++;
                }
            }
            if (userMentions == null && reply2UserIDs == null) {
                relationship = "Tweet";
                csvWriter.writeNext(new String[]{
                        vertex1, vertex1, reciprocated,col3,relationship, dateStr,
                        tweet, urls_and_domains[0], urls_and_domains[1],hashtags, media,
                        tweetImg, dateStr, twitterPage,lat, lon,importedID,
                        inReplyToTweetID, favorited,favoriteCount, inReplyToUserID,isQuoteStatus,
                        language, possiblySensitive,quotedStatusID,retweeted,
                        retweetedCount, retweetID,source,truncated,importedID,
                        importedTweetType
                });
                currLine++;
            }
        }
        return csvWriter;
    }


    private String getTwitterPageForTweet(SolrDocument d) {
        return "https://twitter.com/statuses/"+d.getFieldValue("id");
    }

    private String getEntityHashtags(SolrDocument d) {
        StringBuilder hashtags = new StringBuilder();
        Collection<Object> values = d.getFieldValues("entities_hashtag");
        if (values != null) {
            for (Object o : values) {
                hashtags.append(o.toString()).append(" ");
            }
        }
        return hashtags.toString().trim();
    }

    private String getEntityMedia(SolrDocument d) {
        StringBuilder media = new StringBuilder();
        Collection<Object> values = d.getFieldValues("entities_media_url");
        if (values != null) {
            for (Object o : values) {
                media.append(o.toString()).append(" ");
            }
        }
        return media.toString().trim();
    }

    private String[] getEntityURLsAndDomains(SolrDocument d) {
        Set<String> urls=new HashSet<>();
        Set<String> domains = new HashSet<>();
        Collection<Object> values = d.getFieldValues("entities_url");
        if (values != null) {
            for (Object o : values) {
                try {
                    URL url = Util.expandShortenedURL(o.toString());
                    urls.add(url.toString());
                    domains.add(url.getHost());
                } catch (Exception e) {
                    LOG.warn(ExceptionUtils.getFullStackTrace(e));
                }
            }
        }
        StringBuilder domainStr = new StringBuilder();
        StringBuilder urlStr = new StringBuilder();
        for (String dm : domains)
            domainStr.append(dm).append(" ");
        for (String us : urls)
            urlStr.append(us).append(" ");
        return new String[]{urlStr.toString().trim(), domainStr.toString().trim()};
    }
}
