package SupervisedSRL.Strcutures;

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
    private HashMap<String, Integer> string2intMap;
    private String[] int2stringMap;
    //clusterMap attributes
    public final static int unknownClusterIdx = -100;
    public final static int nullClusterIdx = -200;
    public final static int ROOTClusterIdx = -300;
    //Note clusterMap can not be a HashMap <int, int> as some of the words in the cluster file are not seen in IndexMap
    private HashMap<String, Integer> wordClusterMap;

    public IndexMap(ArrayList<String> trainSentences, String clusterFilePath) throws IOException {
        string2intMap = new HashMap<>();
        string2intMap.put("NULL", nullIdx);
        string2intMap.put("UNK", unknownIdx);
        int index = 2;

        Object[] sets = buildIndividualSets(trainSentences);
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
        HashMap<String, Integer> wordClusterMap  = buildWordClusterMap(clusterFilePath);
        this.wordClusterMap = wordClusterMap;
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

    private HashMap<String, Integer> buildWordClusterMap(String clusterFilePath) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(clusterFilePath)));
        String line2read = "";
        HashMap<String, Integer> wordClusterMap = new HashMap<>();
        HashMap<String, Integer> clusterIDMap = new HashMap<>();

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
        return wordClusterMap;
    }

    public int getClusterId(String str) {
        String word = str;
        int cluster = unknownClusterIdx;
        if (wordClusterMap.containsKey(word))
            cluster = wordClusterMap.get(word);
        return cluster;
    }

}
