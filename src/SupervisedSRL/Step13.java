package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Maryam Aminian on 9/12/16.
 */
public class Step13 {

    public static void evaluate(Properties properties) throws Exception {
        if (!properties.getSteps().contains(13))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 13 -- Evaluation\n>>>>>>>>>>>>>\n");
        HashMap<String, Integer> globalReverseLabelMap = IO.load(properties.getGlobalReverseLabelMapPath());
        String testOutputFile = properties.getOutputFilePathTest_w_projected_info();
        IndexMap indexMap = IO.load(properties.getIndexMapFilePath());
        ArrayList<String> testGoldSentences = IO.readCoNLLFile(properties.getTestFile());
        HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(globalReverseLabelMap);
        reverseLabelMap.put("0", reverseLabelMap.size());
        System.out.println("Evaluating test output >>>>>>\n");
        Evaluation.evaluate(testOutputFile, testGoldSentences, indexMap, reverseLabelMap);
    }
}