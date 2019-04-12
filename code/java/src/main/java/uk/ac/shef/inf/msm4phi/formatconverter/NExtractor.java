package uk.ac.shef.inf.msm4phi.formatconverter;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

public class NExtractor {
    public static void main(String[] args) throws IOException {
        List<String> lines=
                FileUtils.readLines(new File("/home/zz/Cloud/GDrive/ziqizhang/teaching/sheffield/INF6024/2018-19/lab/lab2/named_entities.txt"));
        PrintWriter p = new PrintWriter("/home/zz/Cloud/GDrive/ziqizhang/teaching/sheffield/INF6024/2018-19/lab/lab2/named_entities_.txt");
        for (String l : lines){
            String ne = l.split(",")[1].trim().replaceAll("\\s+","");
            p.println(ne);
        }
        p.close();
    }
}
