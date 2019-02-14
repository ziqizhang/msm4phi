package uk.ac.shef.inf.msm4phi.formatconverter;

import com.opencsv.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;

/**
 * takes nodexl format files and create a csv containing only 1 column of tweets.
 *
 * some params can be used to filter tweets
 */
public class AppTweetBasedFormatAdaptor {
    public static void main(String[] args) throws IOException {
        convert("/home/zz/Cloud/GDrive/ziqizhang/teaching/sheffield/INF6024/2018-19/lab/raw",
                "/home/zz/Cloud/GDrive/ziqizhang/teaching/sheffield/INF6024/2018-19/lab/tweets.txt");
    }

    public static void convert(String inFolder, String outFile) throws IOException {
        File[] files = new File(inFolder).listFiles();
        PrintWriter p = new PrintWriter(outFile);

        Set<String> existing = new HashSet<>();
        for (File f: files){
            Reader reader = Files.newBufferedReader(Paths.get(f.toString()));

            CSVParser parser = new CSVParserBuilder()
                    .withSeparator(',').build();
            CSVReader csvReader =
                    new CSVReaderBuilder(reader)
                            .withSkipLines(1)
                            .withCSVParser(parser)
                            .build();
            String[] nextRecord;
            while ((nextRecord = csvReader.readNext()) != null) {
                String tweet  = nextRecord[6].trim()+".";
                if (tweet.split("\\s+").length<5)
                    continue;

                if (!existing.contains(tweet)){
                    p.println(tweet);
                    existing.add(tweet);
                }
            }
        }
        p.close();
    }
}
