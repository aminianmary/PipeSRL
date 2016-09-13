package SupervisedSRL;

import SupervisedSRL.Reranker.RerankerInstanceGenerator;
import SupervisedSRL.Strcutures.ModelInfo;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step2 {
    public static void buildTrainDataPartitions(ArrayList<String> trainSentences, String partitionsPath, int numOfPartitions) throws Exception {
        RerankerInstanceGenerator rig = new RerankerInstanceGenerator(numOfPartitions);
        ArrayList<String>[] partitions = rig.getPartitions(trainSentences);
        ModelInfo.saveDataPartitions(partitions, partitionsPath);
    }
}
