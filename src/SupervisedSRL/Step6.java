package SupervisedSRL;

import SupervisedSRL.Reranker.Train;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 9/12/16.
 */
public class Step6 {
    public static void Step6 (int numOfPartitions, String instanceFilePrefix, int numOfTrainingIterations,
                              int numOfAIFeatures, int numOfACFeatures, int numOfGlobalFeatures,String rerankerModelPath )
    throws Exception{

        Train.trainReranker(numOfPartitions, instanceFilePrefix, numOfTrainingIterations,
                numOfAIFeatures + numOfACFeatures + numOfGlobalFeatures, rerankerModelPath);
    }
}
