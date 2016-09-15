package SupervisedSRL;

import SupervisedSRL.Strcutures.Properties;

/**
 * Created by Maryam Aminian on 9/12/16.
 */
public class Pipeline {

    //single features 25 + 3 (predicate cluster features) + 5(argument cluster features)
    //p-p features 55
    //a-a feature 91
    //p-a features 154
    //p-a-a features 91
    //some msc tri-gram feature 6
    //joined features based on original paper (ai) 13
    //joined features based on original paper (ac) 15
    //predicate cluster features 3
    //argument cluster features 5

    public final static int numOfPDFeatures = 9;
    public final static int numOfAIFeatures = 25 + 3 + 5 + 13;
    public final static int numOfACFeatures = 25 + 3 + 5 + 13;
    public final static int numOfGlobalFeatures = 1;

    public static void main(String[] args) {
        String trainFile = args[0];
        String devFile = args[1];
        String clusterFile = args[2];
        String modelDir = args[3];
        int numOfPartitions = Integer.parseInt(args[4]);
        int maxNumOfTrainingIterations = Integer.parseInt(args[5]);
        int numOfAIBeamSize = Integer.parseInt(args[6]);
        int numOfACBeamSize = Integer.parseInt(args[7]);
        boolean reranker = Boolean.parseBoolean(args[8]);

        Properties properties = new Properties(trainFile, devFile, clusterFile, modelDir, numOfPartitions,
                maxNumOfTrainingIterations, numOfAIBeamSize, numOfACBeamSize,
                numOfPDFeatures, numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures, reranker);
        try {

            Step1.buildIndexMap(properties);
            Step2.buildTrainDataPartitions(properties);
            Step3.buildModel4EntireData(properties);
            Step3.buildModel4Partitions(properties);
            Step4.buildRerankerFeatureMap(properties);
            Step5.generateRerankerInstances(properties);
            Step6.buildRerankerModel(properties);
            Step7.decode(properties);
            Step8.evaluate(properties);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
