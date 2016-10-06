package Tests;

import SentenceStruct.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.PD.PredicateLexiconEntry;
import SupervisedSRL.Pipeline;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Pair;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.*;

/**
 * Created by monadiab on 8/4/16.
 */
public class FeatureExtractionTest {
    final String tmpFilePath = "/tmp/tmp.tmp";
    final String clusterFilePath = "/tmp/cluster.tmp";
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

    final String clusters = "111101110\tinvented\t11905\n" +
            "111101110\tinaugurated\t9276\n" +
            "111101110\tconsecrated\t8603\n" +
            "111101110\tconceived\t8168\n" +
            "111101110\tconstituted\t7906\n" +
            "111101110\tpatented\t5497\n" +
            "111101110\tdevised\t5143\n" +
            "111101110\tknighted\t4955\n" +
            "111101110\tcoined\t2530\n" +
            "111101110\tplatted\t2241\n" +
            "111101110\tbaptised\t2093\n" +
            "111101110\tpopularized\t2026\n" +
            "111101110\tgazetted\t1877\n" +
            "111101110\trediscovered\t1817\n" +
            "111101110\tconsummated\t1505\n";

    /*
    @Test
    public void testPDFeatures() throws Exception {
        writeConllText();
        writeClusterFile();
        IndexMap map = new IndexMap(tmpFilePath, clusterFilePath);
        int numOfPDFeatures = Pipeline.numOfPDFeatures;
        List<String> textList = new ArrayList<String>();
        textList.add(conllText);

        HashMap<Integer, HashMap<String, HashSet<Object[]>>> lexicon =
                PD.buildPredicateLexicon(textList, map, numOfPDFeatures);

        assert lexicon.containsKey(map.str2int("temperature"));
        assert !lexicon.containsKey(map.str2int("economy"));
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
        writeClusterFile();
        int aiFeatLength = Pipeline.numOfAIFeatures;
        IndexMap map = new IndexMap(tmpFilePath, clusterFilePath);

        List<String> textList = new ArrayList<String>();
        textList.add(conllText);
        Sentence sentence = new Sentence(conllText, map);
        Object[] feats = FeatureExtractor.extractAIFeatures(4, 20, sentence, aiFeatLength, map, false, 0);
        assert feats[3].equals(map.str2int("SBJ"));
        assert feats[17].equals("");


        Object[] feats2 = FeatureExtractor.extractAIFeatures(7, 20, sentence, aiFeatLength, map, false, 0);
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
        assert feats2[14].equals(map.str2int("output"));
        assert feats2[15].equals(map.str2int("NN"));
        assert feats2[16].equals(map.str2int("COORD"));
        String depPath = (map.str2int("ADV") << 1 | 0) + "\t" + (map.str2int("PMOD") << 1 | 0)
                + "\t" + (map.str2int("NMOD") << 1 | 0) + "\t" + (map.str2int("PMOD") << 1 | 0)
                + "\t" + (map.str2int("COORD") << 1 | 0);
        assert feats2[17].equals(depPath);

        assert feats2[19].equals(2);
        assert feats2[20].equals(IndexMap.nullIdx);
        assert feats2[21].equals(IndexMap.nullIdx);
        assert feats2[22].equals(map.str2int("housing"));
        assert feats2[23].equals(map.str2int("NN"));
        assert feats2[24].equals(map.str2int(","));
        assert feats2[25].equals(map.str2int(","));
        assert feats2[26].equals(IndexMap.nullIdx);
        assert feats2[27].equals(IndexMap.nullIdx);
        long expected_pw_aw = map.str2int("taken") << 20 | map.str2int("output");
        assert feats2[33].equals(expected_pw_aw);
        long expected_pw_adeprel = map.str2int("taken") << 10 | map.str2int("COORD");
        assert feats2[35].equals(expected_pw_adeprel);
        String posPath = (map.str2int("IN") << 1 | 0) + "\t" + (map.str2int("NNS") << 1 | 0) + "\t" +
                (map.str2int("IN") << 1 | 0) + "\t" + (map.str2int("NN") << 1 | 0) + "\t" + (0);
        assert feats2[37].equals(map.str2int("taken") + " " + posPath);
        long expected_pw_position = map.str2int("taken") << 2 | 2;
        assert feats2[38].equals(expected_pw_position);
        String expected_deprelpath = (map.str2int("ADV") << 1 | 0) + "\t" + (map.str2int("PMOD") << 1 | 0) + "\t" +
                (map.str2int("NMOD") << 1 | 0) + "\t" + (map.str2int("PMOD") << 1 | 0) + "\t" + (map.str2int("COORD") << 1 | 0);
        assert feats2[64].equals(map.str2int("VC") + " " + expected_deprelpath);

    }


    @Test
    public void testGlobalFeatures() throws Exception {
        writeConllText();
        writeClusterFile();
        List<String> textList = new ArrayList<String>();
        textList.add(conllText);
        ArrayList<Integer> aiCandidIndices = new ArrayList<Integer>();
        ArrayList<Integer> acCandidLabels = new ArrayList<Integer>();
        aiCandidIndices.add(1);
        aiCandidIndices.add(3);
        aiCandidIndices.add(8);
        aiCandidIndices.add(10);
        aiCandidIndices.add(25);
        acCandidLabels.add(1);
        acCandidLabels.add(3);
        acCandidLabels.add(9);
        acCandidLabels.add(4);
        acCandidLabels.add(2);
        Pair<Double, ArrayList<Integer>> aiCandids = new Pair(1.0D, aiCandidIndices);
        Pair<Double, ArrayList<Integer>> acCandids = new Pair(1.0D, acCandidLabels);
        String[] labelMap = {"A0", "A1", "R-A", "A5", "A3", "A4", "A2", "AM", "A-TMP", "A10", "A-VET"};
        Object[] feats = FeatureExtractor.extractGlobalFeatures(7, "take.01", aiCandids, acCandids, labelMap);
        assert feats[0].equals("A1 take.01 A3");

        //predicate seen as the last one
        aiCandidIndices = new ArrayList<Integer>();
        acCandidLabels = new ArrayList<Integer>();
        aiCandidIndices.add(1);
        aiCandidIndices.add(3);
        acCandidLabels.add(1);
        acCandidLabels.add(3);
        aiCandids = new Pair(1.0D, aiCandidIndices);
        acCandids = new Pair(1.0D, acCandidLabels);
        feats = FeatureExtractor.extractGlobalFeatures(7, "take.01", aiCandids, acCandids, labelMap);
        assert feats[0].equals("A1 take.01");

        //predicate seen as the first one
        aiCandidIndices = new ArrayList<Integer>();
        acCandidLabels = new ArrayList<Integer>();
        aiCandidIndices.add(8);
        aiCandidIndices.add(9);
        aiCandidIndices.add(24);
        acCandidLabels.add(2);
        acCandidLabels.add(3);
        acCandidLabels.add(0);
        aiCandids = new Pair(1.0D, aiCandidIndices);
        acCandids = new Pair(1.0D, acCandidLabels);
        feats = FeatureExtractor.extractGlobalFeatures(7, "take.01", aiCandids, acCandids, labelMap);
        assert feats[0].equals("take.01 A0");

        //predicate is the only element
        aiCandidIndices = new ArrayList<Integer>();
        acCandidLabels = new ArrayList<Integer>();
        aiCandidIndices.add(8);
        aiCandidIndices.add(9);
        aiCandidIndices.add(24);
        acCandidLabels.add(2);
        acCandidLabels.add(7);
        acCandidLabels.add(8);
        aiCandids = new Pair(1.0D, aiCandidIndices);
        acCandids = new Pair(1.0D, acCandidLabels);
        feats = FeatureExtractor.extractGlobalFeatures(7, "take.01", aiCandids, acCandids, labelMap);
        assert feats[0].equals("take.01");
    }

    @Test
    public void testAIFeaturesWithGlobalRerankerExtension() throws Exception {
        writeConllText();
        writeClusterFile();
        int aiFeatLength = Pipeline.numOfAIFeatures;
        IndexMap map = new IndexMap(tmpFilePath, clusterFilePath);

        List<String> textList = new ArrayList<String>();
        textList.add(conllText);
        Sentence sentence = new Sentence(conllText, map);
        Object[] feats = FeatureExtractor.extractAIFeatures(4, 20, sentence, aiFeatLength, map, true, 12);
        assert feats[3].equals(map.str2int("SBJ") << 6 | 12);
        assert feats[17].equals(12 + " " + "");


        Object[] feats2 = FeatureExtractor.extractAIFeatures(7, 20, sentence, aiFeatLength, map, true, 24);
        // subcat: ADV, TMP, ADV
        String expectedSubCat = map.str2int("ADV") + "\t" + map.str2int("TMP") + "\t" + map.str2int("ADV");
        assert feats2[7].equals(24 + " " + expectedSubCat);

        TreeSet<Integer> childWordSet = new TreeSet<Integer>();
        childWordSet.add(map.str2int("from"));
        childWordSet.add(map.str2int("week"));
        childWordSet.add(map.str2int("with"));
        String childWordSetStr = "";
        for (int ch : childWordSet)
            childWordSetStr += ch + "\t";
        childWordSetStr = childWordSetStr.trim();
        assert feats2[10].equals(24 + " " + childWordSetStr);
        assert feats2[14].equals(map.str2int("output") << 6 | 24);
        assert feats2[15].equals(map.str2int("NN") << 6 | 24);
        assert feats2[16].equals(map.str2int("COORD") << 6 | 24);
        String depPath = (map.str2int("ADV") << 1 | 0) + "\t" + (map.str2int("PMOD") << 1 | 0)
                + "\t" + (map.str2int("NMOD") << 1 | 0) + "\t" + (map.str2int("PMOD") << 1 | 0)
                + "\t" + (map.str2int("COORD") << 1 | 0);
        assert feats2[17].equals(24 + " " + depPath);

        assert feats2[19].equals(2 << 6 | 24);
        assert feats2[20].equals(IndexMap.nullIdx << 6 | 24);
        assert feats2[21].equals(IndexMap.nullIdx << 6 | 24);
        assert feats2[22].equals(map.str2int("housing") << 6 | 24);
        assert feats2[23].equals(map.str2int("NN") << 6 | 24);
        assert feats2[24].equals(map.str2int(",") << 6 | 24);
        assert feats2[25].equals(map.str2int(",") << 6 | 24);
        assert feats2[26].equals(IndexMap.nullIdx << 6 | 24);
        assert feats2[27].equals(IndexMap.nullIdx << 6 | 24);
        long expected_pw_aw = (map.str2int("taken") << 6 | 24) << 20 | map.str2int("output");
        assert feats2[33].equals(expected_pw_aw);
        long expected_pw_adeprel = (map.str2int("taken") << 6 | 24) << 10 | map.str2int("COORD");
        assert feats2[35].equals(expected_pw_adeprel);
        String posPath = (map.str2int("IN") << 1 | 0) + "\t" + (map.str2int("NNS") << 1 | 0) + "\t" +
                (map.str2int("IN") << 1 | 0) + "\t" + (map.str2int("NN") << 1 | 0) + "\t" + (0);
        assert feats2[37].equals((map.str2int("taken") << 6 | 24) + " " + posPath);
        long expected_pw_position = (map.str2int("taken") << 6 | 24) << 2 | 2;
        assert feats2[38].equals(expected_pw_position);
        String expected_deprelpath = (map.str2int("ADV") << 1 | 0) + "\t" + (map.str2int("PMOD") << 1 | 0) + "\t" +
                (map.str2int("NMOD") << 1 | 0) + "\t" + (map.str2int("PMOD") << 1 | 0) + "\t" + (map.str2int("COORD") << 1 | 0);
        assert feats2[64].equals((map.str2int("VC") << 6 | 24) + " " + expected_deprelpath);
        assert feats2[89].equals("24 take.01 " + map.str2int("output"));

    }

     */
    @Test
    public void testPathFeatures() throws Exception{
        writeConllText();
        writeClusterFile();
        int aiFeatLength = Pipeline.numOfAIFeatures;
        IndexMap map = new IndexMap(tmpFilePath, clusterFilePath);
        Sentence sentence = new Sentence(conllText, map);
        Object[] feats1 = FeatureExtractor.extractAIFeatures(1, 7, sentence, aiFeatLength, map, false, 0);
        Object[] feats2 = FeatureExtractor.extractAIFeatures(6, 11, sentence, aiFeatLength, map, false, 0);
        Object[] feats3 = FeatureExtractor.extractAIFeatures(8, 1, sentence, aiFeatLength, map, false, 0);
        Object[] feats4 = FeatureExtractor.extractAIFeatures(1, 1, sentence, aiFeatLength, map, false, 0);

        String expectedDepPath_1_7 = (map.str2int("NMOD") << 1 | 1) + "\t" + (map.str2int("NMOD") << 1 | 1) + "\t" +
                (map.str2int("SBJ") << 1 | 1) + "\t" + (map.str2int("VC") << 1 | 0) + "\t" + (map.str2int("VC") << 1 | 0);

        String expectedPOSPath_1_7 = (map.str2int("DT") << 1 | 1) + "\t" + (map.str2int("NN") << 1 | 1) + "\t" +
                (map.str2int("NN") << 1 | 1) + "\t" + (map.str2int("MD") << 1 | 0) + "\t" + (map.str2int("VB") << 1 | 0)
                + "\t" + (map.str2int("VBN"));

        String expectedDepPath_6_11 = (map.str2int("VC") << 1 | 0) + "\t" + (map.str2int("ADV") << 1 | 0) + "\t" +
                (map.str2int("PMOD") << 1 | 0);

        String expectedPOSPath_6_11 = (map.str2int("VB") << 1 | 0) + "\t" + (map.str2int("VBN") << 1 | 0) + "\t" +
                (map.str2int("IN") << 1 | 0) + "\t" + (map.str2int("NNS")) ;

        String expectedDepPath_8_1 = (map.str2int("ADV") << 1 | 1) + "\t" + (map.str2int("VC") << 1 | 1) + "\t" +
                (map.str2int("VC") << 1 | 1) + "\t" + (map.str2int("SBJ") << 1 | 0) + "\t" + (map.str2int("NMOD") << 1 | 0)
                + "\t" + (map.str2int("NMOD") << 1 | 0) ;

        String expectedPOSPath_8_1 = (map.str2int("IN") << 1 | 1) + "\t" + (map.str2int("VBN") << 1 | 1) + "\t" +
                (map.str2int("VB") << 1 | 1) + "\t" + (map.str2int("MD") << 1 | 0 ) + "\t" + (map.str2int("NN") << 1 | 0 )
                + "\t" + (map.str2int("NN") << 1 | 0 ) + "\t" + (map.str2int("DT")) ;

        String expectedDepPath_1_1 = "";
        String expectedPOSPath_1_1 = Integer.toString(map.str2int("DT"));

        assert feats1[17].equals(expectedDepPath_1_7);
        assert feats1[18].equals(expectedPOSPath_1_7);
        assert feats2[17].equals(expectedDepPath_6_11);
        assert feats2[18].equals(expectedPOSPath_6_11);
        assert feats3[17].equals(expectedDepPath_8_1);
        assert feats3[18].equals(expectedPOSPath_8_1);
        assert feats4[17].equals(expectedDepPath_1_1);
        assert feats4[18].equals(expectedPOSPath_1_1);
    }

    private void writeConllText() throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter(tmpFilePath));
        writer.write(conllText);
        writer.close();
    }

    private void writeClusterFile() throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter(clusterFilePath));
        writer.write(clusters);
        writer.close();
    }
}
