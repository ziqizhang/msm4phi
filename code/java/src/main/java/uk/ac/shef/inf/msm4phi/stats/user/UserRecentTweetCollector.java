package uk.ac.shef.inf.msm4phi.stats.user;

import com.opencsv.CSVWriter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import twitter4j.Paging;
import twitter4j.ResponseList;
import twitter4j.Status;
import twitter4j.Twitter;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/**
 * collect the recent N tweets from a user's timeline
 *
 * for each user, save in one line, one tweet per column
 *
 * (use api)
 */
public class UserRecentTweetCollector {
    private static final Logger LOG = Logger.getLogger(UserRecentTweetCollector.class.getName());
    private Twitter twitter;
    private int tweetCount;

    public UserRecentTweetCollector(int tweetCount,
                                  String twitterConsumerKey, String twitterConsumerSecret,
                                  String twitterAccessToken, String twitterAccessSecret){
        this.twitter = Util.authenticateTwitter(twitterConsumerKey, twitterConsumerSecret, twitterAccessToken,
                twitterAccessSecret);
        this.tweetCount=tweetCount;
    }

    public void countTweets(List<String[]> users, String outFile) throws IOException {
        Writer writer = Files.newBufferedWriter(Paths.get(outFile));

        CSVWriter csvWriter = new CSVWriter(writer,
                CSVWriter.DEFAULT_SEPARATOR,
                CSVWriter.NO_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END);

        String[] header=new String[3];
        header[0]="user_id";
        header[1]="user_screenname";
        header[2]="recent_n_tweets_ordered_by_time_desc";
        csvWriter.writeNext(header);

        int countU=0;
        for(String[] u : users){
            countU++;

            String screenname=u[1];
            LOG.info(String.format("processing user %s, %s",
                    screenname, String.valueOf(countU)));

            int page=1, total_added=0;

            String[] row =new String[3+tweetCount];
            while(true) {
                LOG.info(String.format("\t\tpage %s",
                        page));
                Paging paging = new Paging(page, tweetCount);
                ResponseList<Status> statuses = null;
                try {
                    statuses = twitter.getUserTimeline(screenname, paging);
                    Thread.sleep(1000);
                } catch (Exception e) {
                    LOG.warn(String.format("\t\tfailed to count user tweets %s \n\t %s, trying again in 10 secs...",
                            screenname, ExceptionUtils.getFullStackTrace(e)));
                    try {
                        Thread.sleep(10000);
                    } catch (InterruptedException e1) {
                    }
                    continue;
                }

                page++;


                for(Status s :statuses){
                    row[total_added+3]=s.getText().replaceAll("\\s+"," ").trim();
                    total_added++;
                    if (total_added==tweetCount)
                        break;
                }

                if(total_added==tweetCount)
                    break;
            }

            csvWriter.writeNext(row);
            csvWriter.flush();
        }


        //save result
        csvWriter.close();
    }
}
