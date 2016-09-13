package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.Pair;
import SupervisedSRL.Strcutures.ProjectConstantPrefixes;
import ml.AveragedPerceptron;
import util.IO;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Maryam Aminian on 9/12/16.
 */
public class Pipeline {
    public static void main(String[] args) {
        String trainFile = args[0];
        String devFile = args[1];
        String clusterFile = args[2];
        String modelDir = args[3];
        int numOfPartitions = Integer.parseInt(args[4]);
        int maxNumOfTrainingIterations = Integer.parseInt(args[5]);
        int numOfAIBeamSize = Integer.parseInt(args[6]);
        int numOfACBeamSize = Integer.parseInt(args[7]);

        String indexMapFilePath = modelDir + ProjectConstantPrefixes.INDEX_MAP;
        String trainDataPartitionsPath = modelDir + ProjectConstantPrefixes.TRAIN_DATA_PARTITIONS;
        String pdModelDir = modelDir;
        String aiModelPath = modelDir + ProjectConstantPrefixes.AI_MODEL;
        String acModelPath = modelDir + ProjectConstantPrefixes.AC_MODEL;
        String partitionPrefix = modelDir + ProjectConstantPrefixes.PARTITION_PREFIX;
        String rerankerFeatureMapPath = modelDir + ProjectConstantPrefixes.RERANKER_FEATURE_MAP;
        String globalReverseLabelMapPath = acModelPath + ProjectConstantPrefixes.GLOBAL_REVERSE_LABEL_MAP;
        String rerankerInstancesPrefix = modelDir + ProjectConstantPrefixes.RERANKER_INSTANCES_PREFIX;
        String rerankerModelPath = modelDir + ProjectConstantPrefixes.RERANKER_MODEL;
        String outputFilePath = modelDir + ProjectConstantPrefixes.OUTPUT_FILE;

        int numOfPDFeatures = 9;
        int numOfAIFeatures = 25;
        int numOfACFeatures = 25;
        int numOfGlobalFeatures = 1;

        try {
            ArrayList<String> trainSentences = IO.readCoNLLFile(trainFile);
            ArrayList<String> devSentences = IO.readCoNLLFile(devFile);

            Step1.buildIndexMap(trainSentences, clusterFile, indexMapFilePath);
            Step2.buildTrainDataPartitions(trainSentences, trainDataPartitionsPath, numOfPartitions);
            Step3.buildModels(trainSentences, devSentences, indexMapFilePath, pdModelDir, aiModelPath, acModelPath,
                    maxNumOfTrainingIterations, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, numOfAIBeamSize, numOfACBeamSize, true);
            ArrayList<String>[] trainDataPartitions = ModelInfo.loadDataPartitions(trainDataPartitionsPath);
            for (int devPartIdx = 0; devPartIdx < trainDataPartitions.length; devPartIdx++) {
                ArrayList<String> trainPart = new ArrayList<String>();
                ArrayList<String> devPart = new ArrayList<String>();
                String pdModelDir4Partition = partitionPrefix + devPartIdx;
                String aiModelPath4Partition = partitionPrefix + devPartIdx + ProjectConstantPrefixes.AI_MODEL;
                String acModelPath4Partition = partitionPrefix + devPartIdx + ProjectConstantPrefixes.AC_MODEL;

                for (int partIdx = 0; partIdx < numOfPartitions; partIdx++) {
                    if (partIdx == devPartIdx)
                        devPart = trainDataPartitions[partIdx];
                    else
                        trainPart.addAll(trainDataPartitions[partIdx]);
                }
                Step3.buildModels(trainPart, devPart, indexMapFilePath, pdModelDir4Partition + devPartIdx, aiModelPath4Partition, acModelPath4Partition,
                        maxNumOfTrainingIterations, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, numOfAIBeamSize, numOfACBeamSize, false);
            }
            Pair<AveragedPerceptron, AveragedPerceptron>[] trainedClassifiersOnPartitions = new Pair[numOfPartitions];
            for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
                String aiModelPath4Partition = partitionPrefix + devPartIdx + ProjectConstantPrefixes.AI_MODEL;
                String acModelPath4Partition = partitionPrefix + devPartIdx + ProjectConstantPrefixes.AC_MODEL;
                trainedClassifiersOnPartitions[devPartIdx] = Step3.loadModels(aiModelPath4Partition, acModelPath4Partition);
            }
            IndexMap indexMap = ModelInfo.loadIndexMap(indexMapFilePath);
            HashMap<String, Integer> globalReverseLabelMap = ModelInfo.loadReverseLabelMap(globalReverseLabelMapPath);
            Step4.buildRerankerFeatureMap(trainedClassifiersOnPartitions, trainDataPartitions, indexMap, globalReverseLabelMap,
                    numOfAIBeamSize, numOfACBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures, numOfGlobalFeatures,
                    partitionPrefix, rerankerFeatureMapPath);

            HashMap<Object, Integer>[] rerankerFeatureMap = ModelInfo.loadFeatureMap(rerankerFeatureMapPath);
            for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
                String pdModelDir4Partition = partitionPrefix + devPartIdx;
                String rerankerInstanceFilePath = rerankerInstancesPrefix + devPartIdx;
                Step5.generateRerankerInstances(trainedClassifiersOnPartitions[devPartIdx], trainDataPartitions[devPartIdx],
                        rerankerFeatureMap, indexMap, globalReverseLabelMap, numOfAIBeamSize, numOfACBeamSize,
                        numOfAIFeatures, numOfACFeatures, numOfPDFeatures, numOfGlobalFeatures,
                        pdModelDir4Partition, rerankerInstanceFilePath);
            }

            Step6.buildRerankerModel(numOfPartitions, rerankerInstancesPrefix, maxNumOfTrainingIterations, numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures, rerankerModelPath);
            Step7.decode(aiModelPath, acModelPath, rerankerModelPath, indexMap, rerankerFeatureMap,
                    devSentences, pdModelDir, outputFilePath, numOfPDFeatures, numOfAIFeatures, numOfACFeatures,
                    numOfAIBeamSize, numOfACBeamSize);
            Step8.evaluate(devSentences, outputFilePath, globalReverseLabelMap, indexMap);

        } catch (Exception e) {
            System.out.print(e.getMessage());
        }
    }
}
