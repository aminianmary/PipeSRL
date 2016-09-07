package Misc;

import util.IO;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by monadiab on 4/26/16.
 */
public class readFrEnEUROPARL {

    public static void main(String[] args) throws IOException {

        String FrFile = args[0];
        String EnFile = args[1];
        String EnFrAlignFile = args[2];
        String GoldFrFile = args[3];

        BufferedReader GoldFrReader = new BufferedReader(new InputStreamReader(new FileInputStream(GoldFrFile)));

        BufferedWriter parallelEnWriter_conll_format = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(EnFile + ".raw.gold")));
        BufferedWriter EnFrWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(EnFrAlignFile + ".gold")));

        BufferedWriter GoldFrCoNllWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(FrFile + ".gold")));
        BufferedWriter GoldEnCoNLLWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(EnFile + ".processed")));

        String goldFRSen = "";
        String gold_line2read = "";
        int goldCounter = -1;

        HashMap<Integer, String> GoldFrSents_index2SenMap = new HashMap<Integer, String>();
        HashMap<String, Integer> GoldFrSents_Sen2EPIndexMap = new HashMap<String, Integer>();

        while ((gold_line2read = GoldFrReader.readLine()) != null) {
            if (gold_line2read.equals("")) {
                goldCounter++;
                GoldFrSents_index2SenMap.put(goldCounter, goldFRSen.trim());
                GoldFrSents_Sen2EPIndexMap.put(goldFRSen.trim(), -1);
                goldFRSen = "";
            } else
                goldFRSen += gold_line2read.trim().split("\t")[1] + " ";
        }

        ///////////////////////////

        Object[] FrObjs = IO.readCoNLLFile_into_words_and_conll_data(FrFile);
        ArrayList<String> FrSents_conll = (ArrayList<String>) FrObjs[0];
        ArrayList<String> FrSents = (ArrayList<String>) FrObjs[1];

        Object[] EnObjs = IO.readCoNLLFile_into_words_and_conll_data(EnFile);
        ArrayList<String> EnSents_conll = (ArrayList<String>) EnObjs[0];
        ArrayList<String> EnSents = (ArrayList<String>) EnObjs[1];

        ArrayList<String> EnFrAlignSents = IO.readPlainFile(EnFrAlignFile);

        for (int idx = 0; idx < FrSents.size(); idx++) {
            String FrSen = FrSents.get(idx);
            if (GoldFrSents_Sen2EPIndexMap.containsKey(FrSen)) {
                if (GoldFrSents_Sen2EPIndexMap.get(FrSen) == -1)
                    GoldFrSents_Sen2EPIndexMap.put(FrSen, idx);
                else
                    System.out.println("SentenceStruct " + idx + " has been observed at least two times!");
            }
        }


        ///////////////////////////
        //writing the prallel En sentences

        for (int goldIdx : GoldFrSents_index2SenMap.keySet()) {

            if (GoldFrSents_Sen2EPIndexMap.get(GoldFrSents_index2SenMap.get(goldIdx)) != -1) {
                int index_in_EP = GoldFrSents_Sen2EPIndexMap.get(GoldFrSents_index2SenMap.get(goldIdx));
                //we had found this FR sentence in EUROPARL
                parallelEnWriter_conll_format.write(IO.formatString2Conll(EnSents.get(index_in_EP).trim()) + "\n");
                EnFrWriter.write(EnFrAlignSents.get(index_in_EP).trim() + "\n");

                GoldFrCoNllWriter.write(FrSents_conll.get(index_in_EP) + "\n");
                GoldEnCoNLLWriter.write(EnSents_conll.get(index_in_EP) + "\n");
            } else {
                parallelEnWriter_conll_format.write("1\tNULL\n\n");
                EnFrWriter.write("\n");
                GoldFrCoNllWriter.write("1\tNULL\n\n");
                GoldEnCoNLLWriter.write("1\tNULL\n\n");
            }
        }

        parallelEnWriter_conll_format.flush();
        parallelEnWriter_conll_format.close();
        EnFrWriter.flush();
        EnFrWriter.close();
        GoldFrCoNllWriter.flush();
        GoldFrCoNllWriter.close();
        GoldEnCoNLLWriter.flush();
        GoldEnCoNLLWriter.close();

    }
}
