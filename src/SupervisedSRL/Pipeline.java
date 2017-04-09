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

    public final static int numOfPIFeatures = 361;
    public final static int numOfPDFeatures = 361;
    public final static int numOfAIFeatures = 202;
    public final static int numOfACFeatures = 202;
    public final static int numOfGlobalFeatures = 1;

    public static void main(String[] args) {
        String trainFile = args[0];
        String devFile = args[1];
        String testFile = args[2];
        String clusterFile = args[3];
        String modelDir = args[4];
        String outputDir = args[5];
        String steps = args[6];
        String modelsToBeTrained = args[7];
        int numOfPartitions = Integer.parseInt(args[8]);
        int maxNumOfPITrainingIterations = Integer.parseInt(args[9]);
        int maxNumOfPDTrainingIterations = Integer.parseInt(args[10]);
        int maxNumOfAITrainingIterations = Integer.parseInt(args[11]);
        int maxNumOfACTrainingIterations = Integer.parseInt(args[12]);
        int maxNumOfRerankerTrainingIterations = Integer.parseInt(args[13]);
        int numOfAIBeamSize = Integer.parseInt(args[14]);
        int numOfACBeamSize = Integer.parseInt(args[15]);
        double aiCoefficient = Double.parseDouble(args[16]);
        boolean reranker = Boolean.parseBoolean(args[17]);
        boolean pi = Boolean.parseBoolean(args[18]);
        boolean supplementOriginalLabels = Boolean.parseBoolean(args[19]);
        String weightedLearning = args[20]; //values: "", "sparse", "dep"


        Properties properties = new Properties(trainFile, devFile, testFile, clusterFile, modelDir, outputDir, numOfPartitions,
                maxNumOfPITrainingIterations, maxNumOfPDTrainingIterations,maxNumOfAITrainingIterations,maxNumOfACTrainingIterations, maxNumOfRerankerTrainingIterations,
                numOfAIBeamSize, numOfACBeamSize, numOfPIFeatures, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures,
                reranker, steps, modelsToBeTrained, aiCoefficient, pi, supplementOriginalLabels, weightedLearning);
        try {
            Step1.buildIndexMap(properties);
            Step2.buildTrainDataPartitions(properties);
            Step3.trainPIModel(properties);
            Step5.trainPDModel(properties);
            Step6.predictPDLabels(properties);
            Step7.trainAIAICModels(properties);
            Step8.buildRerankerFeatureMap(properties);
            Step9.generateRerankerInstances(properties);
            Step10.trainRerankerModel(properties);
            Step11.decode(properties);
            Step12.evaluate(properties);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
