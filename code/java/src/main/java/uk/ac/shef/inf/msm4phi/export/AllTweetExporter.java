package uk.ac.shef.inf.msm4phi.export;


import com.opencsv.CSVReader;
import org.apache.commons.lang.StringEscapeUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.solr.core.CoreContainer;
import org.apache.solr.search.SolrIndexSearcher;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * This class reads the index created by ProdDescExporter, to export product descriptions into batches of txt files
 */
public class AllTweetExporter {

    private static final Logger LOG = Logger.getLogger(AllTweetExporter.class.getName());
    private long maxWordsPerFile = 100000000L;
    //private long maxWordsPerFile=500;
    private int fileCounter = 0;
    private PrintWriter outFile;

    private IndexReader luceneIndexReader;


    public AllTweetExporter(IndexReader prodNameDescIndex) {
        this.luceneIndexReader = prodNameDescIndex;

    }

    public void export(String outFolder, List<String> ignore) {
        long total = 0;
        long resultBatchSize = 50000;
        try {
            outFile = new PrintWriter(new FileWriter(outFolder + "/f_" + fileCounter, true));

            long countFileWords = 0, curr = 0;
            //update results
            System.out.println(String.format("total=%d", luceneIndexReader.maxDoc()));
            for (int i = 0; i < luceneIndexReader.maxDoc(); i++) {

                Document doc = luceneIndexReader.document(i);
                long words = exportRecord(doc, outFile, ignore);
                countFileWords += words;
                if (i % resultBatchSize == 0)
                    LOG.info(String.format("\t\ttotal results of %d, currently processing %d ...",
                            total, i));
                // do something with docId here...

                if (countFileWords >= maxWordsPerFile) {
                    LOG.info(String.format("\t\tfinishing name file, total words= %d",
                            countFileWords));
                    outFile.close();
                    outFile = new PrintWriter(new FileWriter(outFolder + "/f_" + fileCounter, true));
                    countFileWords = 0;
                    fileCounter++;
                }
            }


            try {
                outFile.close();
                System.out.println("Total words="+countFileWords);

            } catch (Exception e) {
                LOG.warn(String.format("\t\tunable to shut down servers due to error: %s",
                        ExceptionUtils.getFullStackTrace(e)));
            }
        } catch (IOException ioe) {
            LOG.warn(String.format("\t\tunable to create output files, io exception: %s",
                    ExceptionUtils.getFullStackTrace(ioe)));
        }
    }

    private long exportRecord(Document d,
                              PrintWriter nameFile, List<String> ignore) {

        String user = d.get("user_screen_name");
        if (ignore.contains(user))
            return 0;
        String tweet = d.get("status_text");


        long tokens = tweet.split("\\s+").length;
        if (tweet.length() > 2) {
            nameFile.println(tweet);
        }

        return tokens;
    }

    public List<String> loadIgnoreList(String inCSV) throws IOException {
        Reader reader = Files.newBufferedReader(Paths.get(inCSV));
        CSVReader csvReader = new CSVReader(reader);
        String[] nextRecord;
        List<String> ignore= new ArrayList<>();
        while ((nextRecord = csvReader.readNext()) != null) {
            ignore.add(nextRecord[14]);
        }
        return ignore;
    }

    public static void main(String[] args) throws IOException {
        CoreContainer coreContainer = new CoreContainer(args[0]);
        coreContainer.load();
        SolrIndexSearcher solrIndexSearcher = coreContainer.getCore("tweets").getSearcher().get();
        IndexReader tweetIndex = solrIndexSearcher.getIndexReader();
        //74488335
        AllTweetExporter exporter = new AllTweetExporter(tweetIndex);

        List<String> ignore = exporter.loadIgnoreList(args[1]);

        exporter.export(args[2], ignore);


        tweetIndex.close();
        System.exit(0);
        LOG.info("COMPLETE!");

    }
}

