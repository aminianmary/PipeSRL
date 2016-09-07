package SupervisedSRL.Strcutures;

import SentStructs.Argument;
import SentStructs.PA;
import SentStructs.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Pipeline;
import SupervisedSRL.Reranker.Train;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 * Created by Maryam Aminian on 6/21/16.
 */
public class IndexMap implements Serializable {
    public final static int nullIdx = 0;
    public final static int unknownIdx = 1;
    private HashMap<String, Integer> string2intMap;
    private String[] int2stringMap;
    private ClusterMap clusterMap;
    int numOfFeatures;
    HashMap<Object, Integer>[] featureMap;
    int numOfPossibleFeatures;
    HashMap<String, Integer> labelMap;

    public IndexMap(ArrayList<String> trainSentences, ClusterMap clusterMap, int numOfFeatures, boolean joint) throws Exception {
        string2intMap = new HashMap<String, Integer>();
        string2intMap.put("NULL", nullIdx);
        string2intMap.put("UNK", unknownIdx);
        int index = 2;
        this.clusterMap = clusterMap;
        this.numOfFeatures = numOfFeatures;

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

        Pair<HashMap<Object, Integer>[], Pair<HashMap<String, Integer>, Integer>> f =  SupervisedSRL.Train.constructFeatureMaps(trainSentences, this, clusterMap, numOfFeatures, joint);
        this.featureMap = f.first;
        this.labelMap = f.second.first;
        this.numOfPossibleFeatures = f.second.second;
    }

    private Object[] buildIndividualSets(String trainFilePath) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(trainFilePath)));
        String line2read = "";

        //data structures to store pos, depRel, words, etc.
        HashSet<String> posTags = new HashSet<String>();
        HashSet<String> depRels = new HashSet<String>();
        HashSet<String> words = new HashSet<String>();

        while ((line2read = reader.readLine()) != null) {
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
        return new Object[]{posTags, depRels, words};
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
}
