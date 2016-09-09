package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step3 {
    public static void main(String[] args) {
        String trainFilePath = args[0]; //train data (either partitions or whole train data) in CoNLL format
        String devFilePath = args[1]; //dev data (either dev partitions (obtained from train data) or real dev data) in CoNLL format
        String indexMapPath = args[2];
        String pdModelDir = args[3];
        String aiModelPath = args[4];
        String acModelPath = args[5];
        int maxTrainingIters = Integer.parseInt(args[6]);
        int numOfPDFeatures = Integer.parseInt(args[7]);
        int numOfAIFeatures = Integer.parseInt(args[8]);
        int numOfACFeatures= Integer.parseInt(args[9]);
        int aiBeamSize = Integer.parseInt(args[10]);
        int acBeamSize = Integer.parseInt(args[11]);
        try{
            IndexMap indexMap = ModelInfo.loadIndexMap(indexMapPath);
            Train.train(trainFilePath, devFilePath, pdModelDir, aiModelPath, acModelPath, indexMap, maxTrainingIters,
                    numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiBeamSize, acBeamSize);
        }catch (Exception e)
        {
            System.out.print(e.getMessage());
        }
    }
}
