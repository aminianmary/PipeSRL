package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step3 {
    public static void Step3(String trainFilePath, String devFilePath, String indexMapPath, String pdModelDir,
                             String aiModelPath, String acModelPath, int maxTrainingIters, int numOfPDFeatures,
                             int numOfAIFeatures, int numOfACFeatures, int aiBeamSize, int acBeamSize) throws Exception{
        IndexMap indexMap = ModelInfo.loadIndexMap(indexMapPath);
        Train.train(trainFilePath, devFilePath, pdModelDir, aiModelPath, acModelPath, indexMap, maxTrainingIters,
                numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiBeamSize, acBeamSize);
    }
}
