package SupervisedSRL.Strcutures;

import util.IO;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by Maryam Aminian on 6/21/16.
 */
public class IndexMap implements Serializable {

    public final static int nullIdx = 0;
    public final static int unknownIdx = 1;
    //clusterMap attributes
    public final static int unknownClusterIdx = -100;
    public final static int nullClusterIdx = -200;
    public final static int ROOTClusterIdx = -300;
    private HashMap<String, Integer> string2intMap;
    private String[] int2stringMap;
    //Note clusterMap can not be a HashMap <int, int> as some of the words in the cluster file are not seen in IndexMap
    private HashMap<String, Integer> wordFullClusterMap;
    private HashMap<Integer, Integer> fullCluster2Cluster4Map;

    public IndexMap(String trainFilePath, String clusterFilePath) throws IOException {
        string2intMap = new HashMap<>();
        string2intMap.put("NULL", nullIdx);
        string2intMap.put("UNK", unknownIdx);
        int index = 2;

        Object[] sets = buildIndividualSets(IO.readCoNLLFile(trainFilePath));
        HashSet<String> posTags = (HashSet<String>) sets[0];
        HashSet<String> depRels = (HashSet<String>) sets[1];
        HashSet<String> words = (HashSet<String>) sets[2];

        for (String posTag : posTags) {
            if (!string2intMap.containsKey(posTag)) {
                string2intMap.put(posTag, index);
                index++;
            }
        }
        for (String depRel : depRels) {
            if (!string2intMap.containsKey(depRel)) {
                string2intMap.put(depRel, index);
                index++;
            }
        }
        for (String word : words) {
            if (!string2intMap.containsKey(word)) {
                string2intMap.put(word, index);
                index++;
            }
        }
        //building int2stringMap
        int2stringMap = new String[string2intMap.size()];
        for (String str : string2intMap.keySet())
            int2stringMap[string2intMap.get(str)] = str;

        //building clusterMap
        Pair<HashMap<String, Integer>, HashMap<Integer, Integer>> clusterMaps = buildWordClusterMap(clusterFilePath);
        this.wordFullClusterMap = clusterMaps.first;
        this.fullCluster2Cluster4Map = clusterMaps.second;
    }

    private Object[] buildIndividualSets(ArrayList<String> trainSentences) {
        //data structures to store pos, depRel, words, etc.
        HashSet<String> posTags = new HashSet<String>();
        HashSet<String> depRels = new HashSet<String>();
        HashSet<String> words = new HashSet<String>();

        for (String sentence : trainSentences) {
            for (String line2read : sentence.split("\n")) {
                if (line2read.equals(""))
                    continue;
                String[] splitLine = line2read.split("\t");
                String id = splitLine[0];
                String form = splitLine[1];
                String gLemma = splitLine[2];
                String pLemma = splitLine[3];
                String gPos = splitLine[4];
                String pPos = splitLine[5];
                String cPos = util.StringUtils.getCoarsePOS(pPos); //coarse predicated pos tag
                String gFeats = splitLine[6];
                String pFeats = splitLine[7];
                String gHead = splitLine[8];
                String pHead = splitLine[9];
                String gDepRel = splitLine[10];
                String pDepRel = splitLine[11];
                String fillPred = splitLine[12];
                String pred = splitLine[13];
                //rest of the splitLine slots are arguments

                posTags.add(gPos);
                posTags.add(pPos);
                posTags.add(cPos);
                depRels.add(gDepRel);
                depRels.add(pDepRel);
                words.add(id);
                words.add(form);
                words.add(gLemma);
                words.add(pLemma);
                for (String gFeat : gFeats.split("|"))
                    words.add(gFeat);
                for (String pFeat : pFeats.split("|"))
                    words.add(pFeat);
                words.add(gHead);
                words.add(pHead);
                words.add(fillPred);
                words.add(pred);
                for (int k = 14; k < splitLine.length; k++)
                    words.add(splitLine[k]);
            }
        }
        return new Object[]{posTags, depRels, words};
    }

    public int str2int(String str) {
        if (string2intMap.containsKey(str))
            return string2intMap.get(str);
        return unknownIdx;
    }

    public String int2str(int i) throws Exception {
        if (int2stringMap.length <= i)
            throw new Exception("Index out of bound");
        return int2stringMap[i];
    }

    private Pair buildWordClusterMap(String clusterFilePath) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(clusterFilePath)));
        String line2read = "";
        HashMap<String, Integer> wordFullClusterIDMap = new HashMap<>();
        HashMap<Integer, Integer> fullClusterID2Cluster4IDMap = new HashMap<>();
        HashMap<String, Integer> fullClusterIDMap = new HashMap<>();
        HashMap<String, Integer> cluster4IDMap = new HashMap<>();

        while ((line2read = reader.readLine()) != null) {
            if (line2read.equals(""))
                continue;
            String[] splitLine = line2read.split("\t");
            String fullClusterBitString = splitLine[0];
            String cluster4BitString = fullClusterBitString.substring(0, Math.min(4, fullClusterBitString.length()));

            if (!fullClusterIDMap.containsKey(fullClusterBitString))
                fullClusterIDMap.put(fullClusterBitString, fullClusterIDMap.size());
            int fullClusterID = fullClusterIDMap.get(fullClusterBitString);

            if (!cluster4IDMap.containsKey(cluster4BitString))
                cluster4IDMap.put(cluster4BitString, cluster4IDMap.size());
            int cluster4ID = cluster4IDMap.get(cluster4BitString);

            String word = splitLine[1];
            wordFullClusterIDMap.put(word, fullClusterID);
            if (!fullClusterID2Cluster4IDMap.containsKey(fullClusterID))
                fullClusterID2Cluster4IDMap.put(fullClusterID, cluster4ID);
        }
        return new Pair<>(wordFullClusterIDMap, fullClusterID2Cluster4IDMap);
    }

    public int getFullClusterId(String str) {
        String word = str;
        int cluster = unknownClusterIdx;
        if (wordFullClusterMap.containsKey(word))
            cluster = wordFullClusterMap.get(word);
        return cluster;
    }

    public int get4ClusterId(String str){
        String word = str;
        int fullClusterID = getFullClusterId(word);
        if (fullClusterID == unknownClusterIdx)
            return unknownClusterIdx;
        else
            return fullCluster2Cluster4Map.get(fullClusterID);
    }

}
