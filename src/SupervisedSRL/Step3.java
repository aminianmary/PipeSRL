package SupervisedSRL;

import SupervisedSRL.PD.PD;
import SupervisedSRL.PI.PI;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 10/21/16.
 */
public class Step3 {

    public static void trainPIModel(Properties properties) throws Exception {
        if (!properties.getSteps().contains(3))
            return;
        buildPIModel4EntireTrainData(properties);
        if (properties.useReranker())
            buildPIModel4TrainPartitions(properties);
    }

    public static void buildPIModel4EntireTrainData(Properties properties) throws Exception{
        if (!properties.getSteps().contains(3))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 3.1 -- Building PI models on entire data\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        String PIModelPath = properties.getPiModelPath();
        String trainFilePath = properties.getTrainFile();
        String devFilePath = properties.getDevFile();
        int maxTrainingIters = properties.getMaxNumOfPITrainingIterations();
        int numOfPIFeatures = properties.getNumOfPIFeatures();
        ArrayList<String> trainSentences = IO.readCoNLLFile(trainFilePath);
        ArrayList<String> devSentences = IO.readCoNLLFile(devFilePath);
        IndexMap indexMap = IO.load(indexMapPath);
        String weightedLearning = properties.isWeightedLearning();

        PI.train(trainSentences, devSentences, indexMap, maxTrainingIters, PIModelPath, numOfPIFeatures, weightedLearning);
    }

    public static void buildPIModel4TrainPartitions(Properties properties) throws Exception {
        if (!properties.getSteps().contains(3) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 3.2 -- Building PI models on partitions\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        int maxTrainingIters = properties.getMaxNumOfPDTrainingIterations();
        int numOfPIFeatures = properties.getNumOfPIFeatures();
        int numOfPartitions = properties.getNumOfPartitions();
        IndexMap indexMap = IO.load(indexMapPath);
        String weightedLearning = properties.isWeightedLearning();

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            System.out.println("\n>>>>>>>>\nPART "+devPartIdx+"\n>>>>>>>>\n");
            String piModelPath = properties.getPartitionPiModelPath(devPartIdx);
            String trainFilePath = properties.getPartitionTrainDataPath(devPartIdx);
            String devFilePath = properties.getPartitionDevDataPath(devPartIdx);
            ArrayList<String> trainSentences = IO.load(trainFilePath);
            ArrayList<String> devSentences = IO.load(devFilePath);

            PI.train(trainSentences, devSentences, indexMap, maxTrainingIters, piModelPath, numOfPIFeatures, weightedLearning);
        }
    }
}
