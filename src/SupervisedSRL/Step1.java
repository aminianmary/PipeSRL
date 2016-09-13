package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 9/9/16.
 * This step creates indexMap for the entire training data and saves it in indexMapFilePath (to be used in the next steps)
 */
public class Step1 {

    public static void buildIndexMap(ArrayList<String> trainData, String clusterFilePath, String indexMapFilePath) throws Exception {
        IndexMap indexMap = new IndexMap(trainData, clusterFilePath);
        ModelInfo.saveIndexMap(indexMap, indexMapFilePath);
    }
}
