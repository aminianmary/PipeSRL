package SupervisedSRL.Strcutures;

import java.io.*;
import java.util.HashMap;

/**
 * Created by monadiab on 9/2/16.
 */
public class ClusterMap implements Serializable {
    public final static int unknownClusterIdx = -100;
    public final static int nullClusterIdx = -200;
    public final static int ROOTClusterIdx = -300;
    private HashMap<String, Integer> wordClusterMap;
    private HashMap<String, Integer> clusterIDMap;

    public ClusterMap(String clusterFilePath) throws IOException {
        Object[] objs = buildWordClusterMap(clusterFilePath);
        HashMap<String, Integer> wordClusterMap = (HashMap<String, Integer>) objs[0];
        HashMap<String, Integer> clusterIDMap = (HashMap<String, Integer>) objs[1];
        this.wordClusterMap = wordClusterMap;
        this.clusterIDMap = clusterIDMap;
    }

    public HashMap<String, Integer> getWordClusterMap() {
        return wordClusterMap;
    }

    public HashMap<String, Integer> getClusterIDMap() {
        return clusterIDMap;
    }

    private Object[] buildWordClusterMap(String clusterFilePath) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(clusterFilePath)));
        String line2read = "";
        HashMap<String, Integer> wordClusterMap = new HashMap<String, Integer>();
        HashMap<String, Integer> clusterIDMap = new HashMap<String, Integer>();

        while ((line2read = reader.readLine()) != null) {
            if (line2read.equals(""))
                continue;
            String[] splitLine = line2read.split("\t");
            String clusterBitString = splitLine[0];
            if (!clusterIDMap.containsKey(clusterBitString))
                clusterIDMap.put(clusterBitString, clusterIDMap.size());
            int clusterID = clusterIDMap.get(clusterBitString);
            String word = splitLine[1];
            wordClusterMap.put(word, clusterID);
        }
        return new Object[]{wordClusterMap, clusterIDMap};
    }

    public int getClusterId(String str) {
        String word = str;
        int cluster = unknownClusterIdx;
        if (wordClusterMap.containsKey(word))
            cluster = wordClusterMap.get(word);
        return cluster;
    }
}
