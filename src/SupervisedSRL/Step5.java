package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step5 {

    public  static void trainAIAICModels(Properties properties) throws Exception {
        if (!properties.getSteps().contains(5))
            return;
        buildModel4EntireData(properties);
        if (properties.useReranker())
            buildModel4Partitions(properties);
    }

    public static void buildModel4EntireData(Properties properties) throws Exception {
        if (!properties.getSteps().contains(5))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 5.1 -- Building AI-AC models on entire data\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        String aiModelPath = properties.getAiModelPath();
        String acModelPath = properties.getAcModelPath();
        String trainFilePath = properties.getTrainFile();
        String devFilePath = properties.getDevFile();
        String trainPDAutoLabelsPath = properties.getTrainAutoPDLabelsPath();
        String devPDAutoLabelsPath = properties.getDevAutoPDLabelsPath();
        int maxAITrainingIters = properties.getMaxNumOfAITrainingIterations();
        int maxACTrainingIters = properties.getMaxNumOfACTrainingIterations();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int aiBeamSize = properties.getNumOfAIBeamSize();
        int acBeamSize = properties.getNumOfACBeamSize();
        double aiCoefficient = properties.getAiCoefficient();
        String modelsToBeTrained = properties.getModelsToBeTrained();

        ArrayList<String> trainSentences = IO.readCoNLLFile(trainFilePath);
        ArrayList<String> devSentences = IO.readCoNLLFile(devFilePath);

        IndexMap indexMap = IO.load(indexMapPath);
        boolean isModelBuiltOnEntireTrainData = true;
        Train.train(trainSentences, devSentences, aiModelPath, acModelPath, indexMap, maxAITrainingIters,maxACTrainingIters,
                numOfAIFeatures, numOfACFeatures, aiBeamSize, acBeamSize, isModelBuiltOnEntireTrainData,
                aiCoefficient, modelsToBeTrained, trainPDAutoLabelsPath, devPDAutoLabelsPath);
    }

    public static void buildModel4Partitions(Properties properties) throws Exception {
        if (!properties.getSteps().contains(5) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 5.2 -- Building AI-AC models on partitions\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        int maxAITrainingIters = properties.getMaxNumOfAITrainingIterations();
        int maxACTrainingIters = properties.getMaxNumOfACTrainingIterations();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int aiBeamSize = properties.getNumOfAIBeamSize();
        int acBeamSize = properties.getNumOfACBeamSize();
        int numOfPartitions = properties.getNumOfPartitions();
        IndexMap indexMap = IO.load(indexMapPath);
        double aiCoefficient = properties.getAiCoefficient();
        String modelsToBeTrained = properties.getModelsToBeTrained();

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            System.out.println("\n>>>>>>>>\nPART "+devPartIdx+"\n>>>>>>>>\n");
            String aiModelPath = properties.getPartitionAIModelPath(devPartIdx);
            String acModelPath = properties.getPartitionACModelPath(devPartIdx);
            String trainFilePath = properties.getPartitionTrainDataPath(devPartIdx);
            String devFilePath = properties.getPartitionDevDataPath(devPartIdx);
            ArrayList<String> trainSentences = IO.load(trainFilePath);
            ArrayList<String> devSentences = IO.load(devFilePath);
            boolean isModelBuiltOnEntireTrainData = false;
            String trainPDAutoLabelsPath = properties.getPartitionTrainPDAutoLabelsPath(devPartIdx);
            String devPDAutoLabelsPath = properties.getPartitionDevPDAutoLabelsPath(devPartIdx);

            Train.train(trainSentences, devSentences, aiModelPath, acModelPath, indexMap, maxAITrainingIters,maxACTrainingIters,
                    numOfAIFeatures, numOfACFeatures, aiBeamSize, acBeamSize, isModelBuiltOnEntireTrainData,
                    aiCoefficient, modelsToBeTrained, trainPDAutoLabelsPath, devPDAutoLabelsPath);
        }
    }

}
