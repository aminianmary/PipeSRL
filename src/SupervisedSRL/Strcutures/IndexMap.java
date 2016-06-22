package SupervisedSRL.Strcutures;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by Maryam Aminian on 6/21/16.
 */
public class IndexMap implements Serializable {

    final int nullIdx = 0;
    final int unknownIdx =1;
    private HashMap<String, Integer> string2intMap;
    private String[] int2stringMap;

    public IndexMap(String trainFilePath) throws IOException
    {
        string2intMap= new HashMap<String, Integer>();
        string2intMap.put("NULL", nullIdx);
        string2intMap.put("UNK",unknownIdx);
        int index =2;

        Object[] sets =buildIndividualSets(trainFilePath);
        HashSet<String> posTags = (HashSet<String>) sets[0];
        HashSet<String> depRels = (HashSet<String>) sets[1];
        HashSet<String> words = (HashSet<String>) sets[2];

        for (String posTag: posTags)
        {
            if (!string2intMap.containsKey(posTag)) {
                string2intMap.put(posTag, index);

            }
        }
        for (String depRel: depRels)
        {
            if (!string2intMap.containsKey(depRel))
                string2intMap.put(depRel, index);
        }
        for (String word: words) {
            if (!string2intMap.containsKey(word))
                string2intMap.put(word, index);
        }
        //building int2stringMap
        int2stringMap= new String[string2intMap.size()];
        for (String str: string2intMap.keySet())
            int2stringMap[string2intMap.get(str)] = str;

    }

    public HashMap<String, Integer> getString2intMap() {return string2intMap;}

    public String[] getInt2stringMap() {return int2stringMap;}

    private Object[] buildIndividualSets (String trainFilePath) throws IOException
    {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(trainFilePath)));
        String line2read= "";

        //data structures to store pos, depRel, words, etc.
        HashSet<String> posTags = new HashSet<String>();
        HashSet<String> depRels = new HashSet<String>();
        HashSet<String> words = new HashSet<String>();

        while ((line2read= reader.readLine())!= null) {
            String[] splitLine= line2read.split("\t");
            String id = splitLine[0];
            String form= splitLine[1];
            String gLemma= splitLine[2];
            String pLemma= splitLine[3];
            String gPos = splitLine[4];
            String pPos = splitLine[5];
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
            depRels.add(gDepRel);
            depRels.add(pDepRel);
            words.add("\t");
            words.add(id);
            words.add(form);
            words.add(gLemma);
            words.add(pLemma);
            words.add("|");
            for (String gFeat: gFeats.split("|"))
                words.add(gFeat);
            for (String pFeat: pFeats.split("|"))
                words.add(pFeat);
            words.add(gHead);
            words.add(pHead);
            words.add(fillPred);
            words.add(pred);
            for (int k=14; k<splitLine.length;k++)
                words.add(splitLine[k]);
        }
        return new Object[]{posTags, depRels, words};
    }

    public int getNullIdx() {return nullIdx;}

    public int getUnknownIdx() {return unknownIdx;}

}
