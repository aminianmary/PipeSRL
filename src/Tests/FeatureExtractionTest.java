package Tests;

import Sentence.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.PD.PredicateLexiconEntry;
import SupervisedSRL.Strcutures.IndexMap;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.*;

/**
 * Created by monadiab on 8/4/16.
 */
public class FeatureExtractionTest {
    final String tmpFilePath = "/tmp/tmp.tmp";
    final String conllText = "1\tThe\tthe\tthe\tDT\tDT\t_\t_\t2\t2\tNMOD\tNMOD\t_\t_\t_\t_\t_\t_\n" +
            "2\teconomy\teconomy\teconomy\tNN\tNN\t_\t_\t4\t4\tNMOD\tNMOD\t_\t_\tA1\t_\t_\t_\n" +
            "3\t's\t's\t's\tPOS\tPOS\t_\t_\t2\t2\tSUFFIX\tSUFFIX\t_\t_\t_\t_\t_\t_\n" +
            "4\ttemperature\ttemperature\ttemperature\tNN\tNN\t_\t_\t5\t5\tSBJ\tSBJ\tY\ttemperature.01\tA2\tA1\t_\t_\n" +
            "5\twill\twill\twill\tMD\tMD\t_\t_\t0\t0\tROOT\tROOT\t_\t_\t_\tAM-MOD\t_\t_\n" +
            "6\tbe\tbe\tbe\tVB\tVB\t_\t_\t5\t5\tVC\tVC\t_\t_\t_\t_\t_\t_\n" +
            "7\ttaken\ttake\ttake\tVBN\tVBN\t_\t_\t6\t6\tVC\tVC\tY\ttake.01\t_\t_\t_\t_\n" +
            "8\tfrom\tfrom\tfrom\tIN\tIN\t_\t_\t7\t7\tADV\tADV\t_\t_\t_\tA2\t_\t_\n" +
            "9\tseveral\tseveral\tseveral\tDT\tDT\t_\t_\t11\t11\tNMOD\tNMOD\t_\t_\t_\t_\t_\t_\n" +
            "10\tvantage\tvantage\tvantage\tNN\tNN\t_\t_\t11\t11\tNMOD\tNMOD\t_\t_\t_\t_\tA1\t_\n" +
            "11\tpoints\tpoint\tpoint\tNNS\tNNS\t_\t_\t8\t8\tPMOD\tPMOD\tY\tpoint.02\t_\t_\t_\t_\n" +
            "12\tthis\tthis\tthis\tDT\tDT\t_\t_\t13\t13\tNMOD\tNMOD\t_\t_\t_\t_\t_\t_\n" +
            "13\tweek\tweek\tweek\tNN\tNN\t_\t_\t7\t7\tTMP\tTMP\t_\t_\t_\tAM-TMP\t_\t_\n" +
            "14\t,\t,\t,\t,\t,\t_\t_\t7\t7\tP\tP\t_\t_\t_\t_\t_\t_\n" +
            "15\twith\twith\twith\tIN\tIN\t_\t_\t7\t7\tADV\tADV\t_\t_\t_\tAM-ADV\t_\t_\n" +
            "16\treadings\treading\treading\tNNS\tNNS\t_\t_\t15\t15\tPMOD\tPMOD\tY\treading.01\t_\t_\t_\t_\n" +
            "17\ton\ton\ton\tIN\tIN\t_\t_\t16\t16\tNMOD\tNMOD\t_\t_\t_\t_\t_\tA1\n" +
            "18\ttrade\ttrade\ttrade\tNN\tNN\t_\t_\t17\t17\tPMOD\tPMOD\t_\t_\t_\t_\t_\t_\n" +
            "19\t,\t,\t,\t,\t,\t_\t_\t18\t18\tP\tP\t_\t_\t_\t_\t_\t_\n" +
            "20\toutput\toutput\toutput\tNN\tNN\t_\t_\t18\t18\tCOORD\tCOORD\t_\t_\t_\t_\t_\t_\n" +
            "21\t,\t,\t,\t,\t,\t_\t_\t20\t20\tP\tP\t_\t_\t_\t_\t_\t_\n" +
            "22\thousing\thousing\thousing\tNN\tNN\t_\t_\t20\t20\tCOORD\tCOORD\t_\t_\t_\t_\t_\t_\n" +
            "23\tand\tand\tand\tCC\tCC\t_\t_\t22\t22\tCOORD\tCOORD\t_\t_\t_\t_\t_\t_\n" +
            "24\tinflation\tinflation\tinflation\tNN\tNN\t_\t_\t23\t23\tCONJ\tCONJ\t_\t_\t_\t_\t_\t_\n" +
            "25\t.\t.\t.\t.\t.\t_\t_\t5\t5\tP\tP\t_\t_\t_\t_\t_\t_\n\n";

    @Test
    public void testPDFeatures() throws Exception {
        writeConllText();
        IndexMap map = new IndexMap(tmpFilePath);

        int numOfPDFeatures = 9;
        List<String> textList = new ArrayList<String>();
        textList.add(conllText);

        HashMap<Integer, HashMap<Integer, HashSet<PredicateLexiconEntry>>> lexicon =
                PD.buildPredicateLexicon(textList, map, numOfPDFeatures);

        assert lexicon.containsKey(map.str2int("temperature"));
        assert !lexicon.containsKey(map.str2int("economy"));
        assert lexicon.get(map.str2int("temperature")).containsKey(map.str2int("NN"));
        assert !lexicon.get(map.str2int("temperature")).containsKey(map.str2int("VB"));
        Object[] feats = ((PredicateLexiconEntry) lexicon.get(map.str2int("temperature")).get(map.str2int("NN")).toArray()[0])
                .getPdfeats();
        assert lexicon.get(map.str2int("temperature")).get(map.str2int("NN")).size() == 1;
        assert feats[3].equals(map.str2int("will"));
        assert feats[0].equals(map.str2int("temperature"));

        feats = ((PredicateLexiconEntry) lexicon.get(map.str2int("take")).get(map.str2int("VB")).toArray()[0])
                .getPdfeats();
        // subcat: ADV, TMP, ADV
        String expectedSubCat = map.str2int("ADV") + "\t" + map.str2int("TMP") + "\t" + map.str2int("ADV");
        assert feats[5].equals(expectedSubCat);

        TreeSet<Integer> childDepSet = new TreeSet<Integer>();
        childDepSet.add(map.str2int("ADV"));
        childDepSet.add(map.str2int("TMP"));
        String childDepSetStr = "";
        for (int ch : childDepSet)
            childDepSetStr += ch + "\t";
        childDepSetStr = childDepSetStr.trim();
        assert feats[6].equals(childDepSetStr);

        TreeSet<Integer> childPOSSet = new TreeSet<Integer>();
        childPOSSet.add(map.str2int("IN"));
        childPOSSet.add(map.str2int("NN"));
        String childPOSSetStr = "";
        for (int ch : childPOSSet)
            childPOSSetStr += ch + "\t";
        childPOSSetStr = childPOSSetStr.trim();
        assert feats[7].equals(childPOSSetStr);

        TreeSet<Integer> childWordSet = new TreeSet<Integer>();
        childWordSet.add(map.str2int("from"));
        childWordSet.add(map.str2int("week"));
        childWordSet.add(map.str2int("with"));
        String childWordSetStr = "";
        for (int ch : childWordSet)
            childWordSetStr += ch + "\t";
        childWordSetStr = childWordSetStr.trim();
        assert feats[8].equals(childWordSetStr);
    }

    @Test
    public void testAIFeatures() throws Exception {
        writeConllText();
        int aiFeatLength = 25 + 13;
        IndexMap map = new IndexMap(tmpFilePath);
        List<String> textList = new ArrayList<String>();
        textList.add(conllText);
        Sentence sentence = new Sentence(conllText, map, false);
        Object[] feats = FeatureExtractor.extractAIFeatures(4, "temperature.01", 20, sentence, aiFeatLength, map);
        assert feats[3].equals(map.str2int("SBJ"));
        assert feats[14].equals("");


        Object[] feats2 = FeatureExtractor.extractAIFeatures(7, "take.01", 20, sentence, aiFeatLength, map);
        // subcat: ADV, TMP, ADV
        String expectedSubCat = map.str2int("ADV") + "\t" + map.str2int("TMP") + "\t" + map.str2int("ADV");
        assert feats2[7].equals(expectedSubCat);

        TreeSet<Integer> childWordSet = new TreeSet<Integer>();
        childWordSet.add(map.str2int("from"));
        childWordSet.add(map.str2int("week"));
        childWordSet.add(map.str2int("with"));
        String childWordSetStr = "";
        for (int ch : childWordSet)
            childWordSetStr += ch + "\t";
        childWordSetStr = childWordSetStr.trim();
        assert feats2[10].equals(childWordSetStr);
        assert feats2[11].equals(map.str2int("output"));
        assert feats2[12].equals(map.str2int("NN"));
        assert feats2[13].equals(map.str2int("COORD"));
        String depPath = (map.str2int("ADV") << 1 | 0) + "\t" + (map.str2int("PMOD") << 1 | 0)
                + "\t" + (map.str2int("NMOD") << 1 | 0) + "\t" + (map.str2int("PMOD") << 1 | 0)
                + "\t" + (map.str2int("COORD") << 1 | 0);
        assert feats2[14].equals(depPath);

        assert feats2[16].equals(2);
        assert feats2[17].equals(IndexMap.nullIdx);
        assert feats2[18].equals(IndexMap.nullIdx);
        assert feats2[19].equals(map.str2int("housing"));
        assert feats2[20].equals(map.str2int("NN"));
        assert feats2[21].equals(map.str2int(","));
        assert feats2[22].equals(map.str2int(","));
        assert feats2[23].equals(IndexMap.nullIdx);
        assert feats2[24].equals(IndexMap.nullIdx);
    }

    private void writeConllText() throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter(tmpFilePath));
        writer.write(conllText);
        writer.close();
    }
}
