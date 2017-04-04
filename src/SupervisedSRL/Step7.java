package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step7 {

    public static void trainAIAICModels(Properties properties) throws Exception {
        if (!properties.getSteps().contains(7))
            return;
        buildModel4EntireData(properties);
        if (properties.useReranker())
            buildModel4Partitions(properties);
    }

    public static void buildModel4EntireData(Properties properties) throws Exception {
        if (!properties.getSteps().contains(7))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 7.1 -- Building AI-AC models on entire data\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        String piModelPath = properties.getPiModelPath();
        String aiModelPath = properties.getAiModelPath();
        String acModelPath = properties.getAcModelPath();
        String trainFilePath = properties.getTrainFile();
        String devFilePath = properties.getDevFile();
        String trainPDAutoLabelsPath = properties.getTrainPDLabelsPath();
        String pdModelDir = properties.getPdModelDir();
        int maxAITrainingIters = properties.getMaxNumOfAITrainingIterations();
        int maxACTrainingIters = properties.getMaxNumOfACTrainingIterations();
        int numOfPIFeatures = properties.getNumOfPIFeatures();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int aiBeamSize = properties.getNumOfAIBeamSize();
        int acBeamSize = properties.getNumOfACBeamSize();
        double aiCoefficient = properties.getAiCoefficient();
        String modelsToBeTrained = properties.getModelsToBeTrained();
        boolean usePI = properties.usePI();
        boolean supplement = properties.supplementOriginalLabels();
        boolean weightedLearning = properties.isWeightedLearning();

        ArrayList<String> trainSentences = IO.readCoNLLFile(trainFilePath);
        ArrayList<String> devSentences = IO.readCoNLLFile(devFilePath);

        IndexMap indexMap = IO.load(indexMapPath);
        boolean isModelBuiltOnEntireTrainData = true;
        Train.train(trainSentences, devSentences, piModelPath, aiModelPath, acModelPath, indexMap, maxAITrainingIters, maxACTrainingIters,
                numOfPIFeatures, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, aiBeamSize, acBeamSize, isModelBuiltOnEntireTrainData,
                aiCoefficient, modelsToBeTrained, trainPDAutoLabelsPath, pdModelDir, usePI, supplement, weightedLearning);
    }

    public static void buildModel4Partitions(Properties properties) throws Exception {
        if (!properties.getSteps().contains(7) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 7.2 -- Building AI-AC models on partitions\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        int maxAITrainingIters = properties.getMaxNumOfAITrainingIterations();
        int maxACTrainingIters = properties.getMaxNumOfACTrainingIterations();
        int numOfPIFeatures = properties.getNumOfPIFeatures();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int aiBeamSize = properties.getNumOfAIBeamSize();
        int acBeamSize = properties.getNumOfACBeamSize();
        int numOfPartitions = properties.getNumOfPartitions();
        IndexMap indexMap = IO.load(indexMapPath);
        double aiCoefficient = properties.getAiCoefficient();
        String modelsToBeTrained = properties.getModelsToBeTrained();
        boolean usePI = properties.usePI();
        boolean supplement = properties.supplementOriginalLabels();
        boolean weightedLearning = properties.isWeightedLearning();

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            System.out.println("\n>>>>>>>>\nPART " + devPartIdx + "\n>>>>>>>>\n");
            String piModelPath = properties.getPartitionPiModelPath(devPartIdx);
            String aiModelPath = properties.getPartitionAIModelPath(devPartIdx);
            String acModelPath = properties.getPartitionACModelPath(devPartIdx);
            String trainFilePath = properties.getPartitionTrainDataPath(devPartIdx);
            String devFilePath = properties.getPartitionDevDataPath(devPartIdx);
            String pdModelDir = properties.getPartitionPdModelDir(devPartIdx);
            ArrayList<String> trainSentences = IO.load(trainFilePath);
            ArrayList<String> devSentences = IO.load(devFilePath);
            boolean isModelBuiltOnEntireTrainData = false;
            String trainPDAutoLabelsPath = properties.getPartitionTrainPDAutoLabelsPath(devPartIdx);

            Train.train(trainSentences, devSentences, piModelPath, aiModelPath, acModelPath, indexMap, maxAITrainingIters, maxACTrainingIters,
                    numOfPIFeatures, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, aiBeamSize, acBeamSize, isModelBuiltOnEntireTrainData,
                    aiCoefficient, modelsToBeTrained, trainPDAutoLabelsPath, pdModelDir, usePI, supplement, weightedLearning);
        }
    }

}
