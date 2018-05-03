package uk.ac.shef.inf.msm4phi;

import com.opencsv.CSVWriter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;

import java.io.IOException;
import java.net.URL;
import java.util.*;
import java.util.concurrent.RecursiveTask;

public class WorkerTweetExport extends RecursiveTask<Integer> {
    private SolrClient solrClient;
    private int resultBatchSize = 5000;
    private String outFolder;
    private int maxTasksPerThread;
    private static final Logger LOG = Logger.getLogger(WorkerTweetExport.class.getName());

    private Map<String, List<String>> hashtagMap;

    public WorkerTweetExport(SolrClient solrClient, Map<String, List<String>> hashtagMap, int maxTasksPerThread,
                             String outFolder) {
        this.solrClient = solrClient;
        this.hashtagMap = hashtagMap;
        this.maxTasksPerThread = maxTasksPerThread;
        this.outFolder = outFolder;
    }

    public int getResultBatchSize() {
        return this.resultBatchSize;
    }

    public void setResultBatchSize(int resultBatchSize) {
        this.resultBatchSize = resultBatchSize;
    }

    @Override
    protected Integer compute() {
        if (this.hashtagMap.size() > maxTasksPerThread) {
            List<WorkerTweetExport> subWorkers =
                    new ArrayList<>(createSubWorkers());
            for (WorkerTweetExport subWorker : subWorkers)
                subWorker.fork();
            return mergeResult(subWorkers);
        } else {
            return computeSingleWorker(hashtagMap);
        }
    }


    /**
     * Query the solr backend to process tweets
     *
     * @param tasks
     * @return
     */
    protected int computeSingleWorker(Map<String, List<String>> tasks) {
        for (Map.Entry<String, List<String>> en : tasks.entrySet()) {
            LOG.info(String.format("\t processing hashtag '%s' with %d variants...", en.getKey(), en.getValue().size()));
            SolrQuery q = Util.createQueryTweetsOfHashtags(resultBatchSize, en.getValue().toArray(new String[0]));

            CSVWriter csvWriter = null;
            try {
                csvWriter = Util.createCSVWriter(outFolder + "/" + en.getKey().replaceAll("#", "_"));
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
                    LOG.info(String.format("\ttotal results of %d, currently processing from %d to %d...",
                            total, q.getStart(), q.getStart() + q.getRows()));
                    writeCSVContent(csvWriter, res.getResults());

                } catch (Exception e) {
                    LOG.warn(String.format("\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                            q.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
                }

                int curr = q.getStart() + q.getRows();
                if (curr < total)
                    q.setStart(curr);
                else
                    stop = true;

            }
            if (csvWriter!=null) {
                try {
                    csvWriter.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return 0;
    }


    protected List<WorkerTweetExport> createSubWorkers() {
        List<WorkerTweetExport> subWorkers =
                new ArrayList<>();

        boolean b = false;
        Map<String, List<String>> splitTask1 = new HashMap<>();
        Map<String, List<String>> splitTask2 = new HashMap<>();
        for (Map.Entry<String, List<String>> e : hashtagMap.entrySet()) {
            if (b)
                splitTask1.put(e.getKey(), e.getValue());
            else
                splitTask2.put(e.getKey(), e.getValue());
            b = !b;
        }

        WorkerTweetExport subWorker1 = createInstance(splitTask1);
        WorkerTweetExport subWorker2 = createInstance(splitTask2);

        subWorkers.add(subWorker1);
        subWorkers.add(subWorker2);

        return subWorkers;
    }

    protected WorkerTweetExport createInstance(Map<String, List<String>> splitTasks) {
        return new WorkerTweetExport(this.solrClient, splitTasks, maxTasksPerThread, outFolder);
    }

    protected int mergeResult(List<WorkerTweetExport> workers) {
        Integer total = 0;
        for (WorkerTweetExport worker : workers) {
            total += worker.join();
        }
        return total;
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
    private void writeCSVContent(CSVWriter csvWriter, SolrDocumentList results) {
        for (SolrDocument d : results) {
            String vertex1 = d.getFieldValue("user_screen_name").toString();
            String reciprocated = "";
            String col3 = "";
            String relationship = "";
            Date relationshipDate = (Date) d.getFieldValue("created_at");//todo
            String dateStr="";
            String tweet = d.getFieldValue("status_text").toString();
            String[] urls_and_domains = getEntityURLsAndDomains(d);
            String hashtags = getEntityHashtags(d);
            String media=getEntityMedia(d);
            String tweetImg="";
            String twitterPage=getTwitterPageForTweet(d);
            String lat = d.getFieldValue("coordinate_lat")==null?"":d.getFieldValue("coordinate_lat").toString();
            String lon = d.getFieldValue("coordinate_lon")==null?"":d.getFieldValue("coordinate_lon").toString();
            String importedID=d.getFieldValue("id").toString();
            String inReplyToTweetID=d.getFirstValue("in_reply_to_status_id_str")==null?"":
                    d.getFirstValue("in_reply_to_status_id_str").toString();
            String favorited="0";
            String favoriteCount=d.getFirstValue("favorite_count")==null?"":
                    d.getFirstValue("favorite_count").toString();
            String inReplyToUserID=d.getFirstValue("in_reply_to_screen_name")==null?"":
                    d.getFirstValue("in_reply_to_screen_name").toString();
            String isQuoteStatus=d.getFirstValue("is_quote_status")==null?"":
                    d.getFirstValue("is_quote_status").toString();
            String language="en";
            String possiblySensitive="";
            String quotedStatusID=d.getFirstValue("quoted_status_id_str")==null?"":
                    d.getFirstValue("quoted_status_id_str").toString();
            String retweeted="0";
            String retweetedCount=d.getFirstValue("retweet_count")==null?"":
                    d.getFirstValue("retweet_count").toString();
            String retweetID=d.getFirstValue("retweeted_status_id_str")==null?"":
                    d.getFirstValue("retweeted_status_id_str").toString();
            String source="", truncated="", importedTweetType="";


            Collection<Object> userMentions = d.getFieldValues("entities_user_mention");
            if (userMentions != null) {
                relationship = "Mentions";
                for(Object vertex2 : userMentions) {
                    csvWriter.writeNext(new String[]{
                            vertex1, vertex2.toString(), reciprocated,col3,relationship, dateStr,
                            tweet, urls_and_domains[0], urls_and_domains[1],hashtags, media,
                            tweetImg, twitterPage,lat, lon,importedID,
                            inReplyToTweetID, favorited,favoriteCount, inReplyToUserID,isQuoteStatus,
                            language, possiblySensitive,quotedStatusID,retweeted,
                            retweetedCount, retweetID,source,truncated,importedID,
                            importedTweetType
                    });
                }
            }
            Collection<Object> reply2UserIDs = d.getFieldValues("in_reply_to_screen_name");
            if (reply2UserIDs != null) {
                relationship = "Replies to";
                for(Object vertex2 : reply2UserIDs) {
                    csvWriter.writeNext(new String[]{
                            vertex1, vertex2.toString(), reciprocated,col3,relationship, dateStr,
                            tweet, urls_and_domains[0], urls_and_domains[1],hashtags, media,
                            tweetImg, twitterPage,lat, lon,importedID,
                            inReplyToTweetID, favorited,favoriteCount, inReplyToUserID,isQuoteStatus,
                            language, possiblySensitive,quotedStatusID,retweeted,
                            retweetedCount, retweetID,source,truncated,importedID,
                            importedTweetType
                    });
                }
            }
            if (userMentions == null && reply2UserIDs == null) {
                relationship = "Tweet";
                csvWriter.writeNext(new String[]{
                        vertex1, vertex1, reciprocated,col3,relationship, dateStr,
                        tweet, urls_and_domains[0], urls_and_domains[1],hashtags, media,
                        tweetImg, twitterPage,lat, lon,importedID,
                        inReplyToTweetID, favorited,favoriteCount, inReplyToUserID,isQuoteStatus,
                        language, possiblySensitive,quotedStatusID,retweeted,
                        retweetedCount, retweetID,source,truncated,importedID,
                        importedTweetType
                });
            }
        }
    }

    private String getTwitterPageForTweet(SolrDocument d) {
        return "https://twitter.com/statuses/"+d.getFieldValue("id");
    }

    private String getEntityHashtags(SolrDocument d) {
        StringBuilder hashtags = new StringBuilder();
        Collection<Object> values = d.getFieldValues("entities_hashtag");
        if (values != null) {
            for (Object o : values) {
                hashtags.append(hashtags.toString()).append(" ");
            }
        }
        return hashtags.toString();
    }

    private String getEntityMedia(SolrDocument d) {
        StringBuilder media = new StringBuilder();
        Collection<Object> values = d.getFieldValues("entities_media_url");
        if (values != null) {
            for (Object o : values) {
                media.append(media.toString()).append(" ");
            }
        }
        return media.toString();
    }

    private String[] getEntityURLsAndDomains(SolrDocument d) {
        StringBuilder urls = new StringBuilder();
        Set<String> domains = new HashSet<>();
        Collection<Object> values = d.getFieldValues("entities_url");
        if (values != null) {
            for (Object o : values) {
                try {
                    URL url = Util.expandShortenedURL(o.toString());
                    urls.append(url.toString()).append(" ");
                    domains.add(url.getHost());
                } catch (Exception e) {
                    LOG.warn(ExceptionUtils.getFullStackTrace(e));
                }
            }
        }
        StringBuilder domainStr = new StringBuilder();
        for (String dm : domains) {
            domainStr.append(dm).append(" ");
        }
        return new String[]{urls.toString().trim(), domainStr.toString().trim()};
    }
}