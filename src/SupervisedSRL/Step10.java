package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Maryam Aminian on 9/12/16.
 */
public class Step10 {

    public static void evaluate(Properties properties) throws Exception {
        if (!properties.getSteps().contains(10))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 10 -- Evaluation\n>>>>>>>>>>>>>\n");
        HashMap<String, Integer> globalReverseLabelMap = IO.load(properties.getGlobalReverseLabelMapPath());
        String outputFile = properties.getOutputFilePath();
        IndexMap indexMap = IO.load(properties.getIndexMapFilePath());
        ArrayList<String> goldSentences = IO.readCoNLLFile(properties.getDevFile());
        HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(globalReverseLabelMap);
        reverseLabelMap.put("0", reverseLabelMap.size());
        Evaluation.evaluate(outputFile, goldSentences, indexMap, reverseLabelMap);
    }
}
