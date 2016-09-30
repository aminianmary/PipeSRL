package Tests;

import SentenceStruct.*;
import SupervisedSRL.Strcutures.IndexMap;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.TreeSet;

/**
 * Created by monadiab on 8/3/16.
 */
public class SentenceTest {
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

    @Test
    public void testConstructor() throws Exception {
        writeConllText();
        writeClusterFile();
        IndexMap map = new IndexMap(tmpFilePath, clusterFilePath);
        Sentence sentence = new Sentence(conllText, map);

        int[] depHeads = sentence.getDepHeads();
        assert depHeads[0] == 0;
        assert depHeads[3] == 2;

        int[] depLabels = sentence.getDepLabels();
        assert depLabels[3] == map.str2int("SUFFIX");
        assert depLabels[0] == IndexMap.nullIdx;

        int[] words = sentence.getWords();
        assert words[11] == map.str2int("points");

        int[] posTags = sentence.getPosTags();
        assert posTags[10] == map.str2int("NN");

        int[] cPOSTags = sentence.getcPosTags();
        assert cPOSTags[17] == map.str2int("N");

        int[] lemmas = sentence.getLemmas();
        assert lemmas[0] == map.str2int("ROOT");
        assert lemmas[8] == map.str2int("from");

        TreeSet<Integer>[] reverseHead = sentence.getReverseDepHeads();
        assert reverseHead[2].size() == 2;
        assert reverseHead[2].contains(3);

        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa : pas) {
            System.out.println(pa.getPredicate().getIndex() + "\t" + pa.getPredicate().getPredicateGoldLabel());
            if (pa.getPredicate().getIndex() == 4) {
                assert pa.getPredicate().getPredicateGoldLabel().equals("temperature.01");
                assert pa.getAllPredArgDepTupls().contains(new PADependencyTuple(4, 4, "A2"));
                assert pa.obtainArgumentType(4).equals("A2");
                assert pa.obtainArgumentType(2).equals("A1");
                for (Argument arg : pa.getArguments()) {
                    if (arg.getIndex() == 4)
                        assert arg.getArgPosition() == ArgumentPosition.ON;
                    if (arg.getIndex() == 2)
                        assert arg.getArgPosition() == ArgumentPosition.BEFORE;
                }
            }
            for (Argument arg : pa.getArguments()) {
                System.out.println("arg: " + arg.getIndex() + "\t" + arg.getType());
                if (pa.getPredicate().getIndex() == 16)
                    assert arg.getIndex() == 17;
            }
        }

    }

    @Test
    public void testGetDepPath() throws Exception {
        writeConllText();
        writeClusterFile();
        IndexMap map = new IndexMap(tmpFilePath, clusterFilePath);
        Sentence sentence = new Sentence(conllText, map);
        ArrayList<Integer> depPath = sentence.getDepPath(5, 12);

        assert depPath.get(0) == (map.str2int("VC") << 1 | 0);
        assert depPath.get(3) == (map.str2int("NMOD") << 1 | 1);
    }

    @Test
    public void testGetPOSPath() throws Exception {
        writeConllText();
        writeClusterFile();
        IndexMap map = new IndexMap(tmpFilePath, clusterFilePath);
        Sentence sentence = new Sentence(conllText, map);
        ArrayList<Integer> depPath = sentence.getPOSPath(5, 12);

        assert depPath.get(0) == (map.str2int("VB") << 1 | 0);
        assert depPath.get(3) == 1;
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
