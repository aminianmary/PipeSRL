package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;

import java.io.IOException;
import java.util.HashMap;

/**
 * Created by monadiab on 9/12/16.
 */
public class Step8 {

    public static void Step8(String goldData, String outputFile, HashMap<String, Integer> globalReverseLabelMap,
                             IndexMap indexMap) throws IOException {
        HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(globalReverseLabelMap);
        reverseLabelMap.put("0", reverseLabelMap.size());
        Evaluation.evaluate(outputFile, goldData, indexMap, reverseLabelMap);
    }
}
