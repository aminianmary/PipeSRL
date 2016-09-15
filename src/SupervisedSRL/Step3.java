package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step3 {
    public static void buildModel4EntireData(Properties properties) throws Exception {
        if (!properties.getSteps().contains(3))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 3.1 -- Building PD-AI-AC models on entire data\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        String pdModelDir = properties.getPdModelDir();
        String aiModelPath = properties.getAiModelPath();
        String acModelPath = properties.getAcModelPath();
        String trainFilePath = properties.getTrainFile();
        String devFilePath = properties.getDevFile();
        int maxTrainingIters = properties.getMaxNumOfTrainingIterations();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int aiBeamSize = properties.getNumOfAIBeamSize();
        int acBeamSize = properties.getNumOfACBeamSize();

        ArrayList<String> trainSentences = IO.readCoNLLFile(trainFilePath);
        ArrayList<String> devSentences = IO.readCoNLLFile(devFilePath);

        IndexMap indexMap = IO.load(indexMapPath);
        boolean isModelBuiltOnEntireTrainData = true;
        Train.train(trainSentences, devSentences, pdModelDir, aiModelPath, acModelPath, indexMap, maxTrainingIters,
                numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiBeamSize, acBeamSize, isModelBuiltOnEntireTrainData);
    }

    public static void buildModel4Partitions(Properties properties) throws Exception {
        if (!properties.getSteps().contains(3) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 3.2 -- Building PD-AI-AC models on partitions\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        int maxTrainingIters = properties.getMaxNumOfTrainingIterations();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int aiBeamSize = properties.getNumOfAIBeamSize();
        int acBeamSize = properties.getNumOfACBeamSize();
        int numOfPartitions = properties.getNumOfPartitions();
        IndexMap indexMap = IO.load(indexMapPath);

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            System.out.println("\n>>>>>>>>\nPART "+devPartIdx+"\n>>>>>>>>\n");
            String pdModelDir = properties.getPartitionPdModelDir(devPartIdx);
            String aiModelPath = properties.getPartitionAIModelPath(devPartIdx);
            String acModelPath = properties.getPartitionACModelPath(devPartIdx);
            String trainFilePath = properties.getPartitionTrainDataPath(devPartIdx);
            String devFilePath = properties.getPartitionDevDataPath(devPartIdx);
            ArrayList<String> trainSentences = IO.load(trainFilePath);
            ArrayList<String> devSentences = IO.load(devFilePath);
            boolean isModelBuiltOnEntireTrainData = false;
            Train.train(trainSentences, devSentences, pdModelDir, aiModelPath, acModelPath, indexMap, maxTrainingIters,
                    numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiBeamSize, acBeamSize, isModelBuiltOnEntireTrainData);
        }
    }

}
