package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by monadiab on 9/12/16.
 */
public class Step8 {

    public static void evaluate(Properties properties) throws Exception {
        HashMap<String, Integer> globalReverseLabelMap = ModelInfo.loadReverseLabelMap(properties.getGlobalReverseLabelMapPath());
        String outputFile = properties.getOutputFilePath();
        IndexMap indexMap = ModelInfo.loadIndexMap(properties.getIndexMapFilePath());
        ArrayList<String> goldSentences= IO.readCoNLLFile(properties.getDevFile());
        HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(globalReverseLabelMap);
        reverseLabelMap.put("0", reverseLabelMap.size());
        Evaluation.evaluate(outputFile, goldSentences, indexMap, reverseLabelMap);
    }
}
