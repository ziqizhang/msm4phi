package uk.ac.shef.inf.msm4phi.formatconverter;

import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * takes file output for nodexl and convert them to cytoscape required
 */
public class AppCytoscapeFormatAdaptor {
    public static void main(String[] args) throws IOException {
        convert("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/nodexl/manU",
                "/home/zz/Cloud/GDrive/ziqizhang/teaching/sheffield/INF6024/2018-19/lab/edges.csv");
    }

    public static void convert(String inFolder, String outFile) throws IOException {
        File[] files = new File(inFolder).listFiles();

        Map<String, Integer> weights = new HashMap<>();
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
                String user1  = nextRecord[0].trim();
                String user2 = nextRecord[1].trim();
                String key = user1+"|"+user2;
                if (weights.containsKey(key))
                    weights.put(key,weights.get(key)+1);
                else{
                    weights.put(key,1);
                }
            }
        }

        PrintWriter p = new PrintWriter(outFile);
        p.println("From, To, Weight");
        for(Map.Entry<String, Integer> e: weights.entrySet()) {
            String[] users = e.getKey().split("\\|");
            p.println(users[0] + "," + users[1] + "," + e.getValue());
        }
        p.close();
    }
}
