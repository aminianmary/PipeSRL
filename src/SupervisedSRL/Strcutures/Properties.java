package SupervisedSRL.Strcutures;

/**
 * Created by Maryam Aminian on 9/13/16.
 */

public class Properties {
    private String trainFile;
    private String devFile;
    private String clusterFile;
    private String modelDir;
    private int numOfPartitions;
    private int maxNumOfTrainingIterations;
    private int numOfAIBeamSize;
    private int numOfACBeamSize;
    private int numOfPDFeatures;
    private int numOfAIFeatures;
    private int numOfACFeatures;
    private int numOfGlobalFeatures;
    private String indexMapFilePath;
    private String pdModelDir;
    private String aiModelPath;
    private String acModelPath;
    private String partitionPrefix;
    private String rerankerFeatureMapPath;
    private String globalReverseLabelMapPath;
    private String rerankerModelPath;
    private String outputFilePath;

    public Properties(String trainFile, String devFile, String clusterFile, String modelDir,
                      int numOfPartitions, int maxNumOfTrainingIterations, int numOfAIBeamSize, int numOfACBeamSize,
                      int numOfPDFeatures, int numOfAIFeatures, int numOfACFeatures, int numOfGlobalFeatures) {
        this.numOfGlobalFeatures = numOfGlobalFeatures;
        this.trainFile = trainFile;
        this.devFile = devFile;
        this.clusterFile = clusterFile;
        this.modelDir = modelDir;
        this.numOfPartitions = numOfPartitions;
        this.maxNumOfTrainingIterations = maxNumOfTrainingIterations;
        this.numOfAIBeamSize = numOfAIBeamSize;
        this.numOfACBeamSize = numOfACBeamSize;
        this.numOfPDFeatures = numOfPDFeatures;
        this.numOfAIFeatures = numOfAIFeatures;
        this.numOfACFeatures = numOfACFeatures;
        this.indexMapFilePath = modelDir + ProjectConstantPrefixes.INDEX_MAP;
        this.pdModelDir = modelDir;
        this.aiModelPath = modelDir + ProjectConstantPrefixes.AI_MODEL;
        this.acModelPath = modelDir + ProjectConstantPrefixes.AC_MODEL;
        this.partitionPrefix = modelDir + ProjectConstantPrefixes.PARTITION_PREFIX;
        this.rerankerFeatureMapPath = modelDir + ProjectConstantPrefixes.RERANKER_FEATURE_MAP;
        this.globalReverseLabelMapPath = acModelPath + ProjectConstantPrefixes.GLOBAL_REVERSE_LABEL_MAP;
        this.rerankerModelPath = modelDir + ProjectConstantPrefixes.RERANKER_MODEL;
        this.outputFilePath = modelDir + ProjectConstantPrefixes.OUTPUT_FILE;
    }

    public int getNumOfGlobalFeatures() {
        return numOfGlobalFeatures;
    }

    public String getTrainFile() {
        return trainFile;
    }

    public String getDevFile() {
        return devFile;
    }

    public String getClusterFile() {
        return clusterFile;
    }

    public String getModelDir() {
        return modelDir;
    }

    public int getNumOfPartitions() {
        return numOfPartitions;
    }

    public int getMaxNumOfTrainingIterations() {
        return maxNumOfTrainingIterations;
    }

    public int getNumOfAIBeamSize() {
        return numOfAIBeamSize;
    }

    public int getNumOfACBeamSize() {
        return numOfACBeamSize;
    }

    public int getNumOfPDFeatures() {
        return numOfPDFeatures;
    }

    public int getNumOfAIFeatures() {
        return numOfAIFeatures;
    }

    public int getNumOfACFeatures() {
        return numOfACFeatures;
    }

    public String getIndexMapFilePath() {
        return indexMapFilePath;
    }

    public String getPdModelDir() {
        return pdModelDir;
    }

    public String getAiModelPath() {
        return aiModelPath;
    }

    public String getAcModelPath() {
        return acModelPath;
    }

    public String getPartitionPrefix() {
        return partitionPrefix;
    }

    public String getRerankerFeatureMapPath() {
        return rerankerFeatureMapPath;
    }

    public String getGlobalReverseLabelMapPath() {
        return globalReverseLabelMapPath;
    }

    public String getRerankerModelPath() {
        return rerankerModelPath;
    }

    public String getOutputFilePath() {
        return outputFilePath;
    }

    public String getPartitionTrainDataPath(int devPartIdx) {return  partitionPrefix + devPartIdx + ProjectConstantPrefixes.PARTITION_TRAIN_DATA;}

    public String getPartitionDevDataPath(int devPartIdx) {return  partitionPrefix + devPartIdx + ProjectConstantPrefixes.PARTITION_DEV_DATA;}

    public String getPartitionPdModelDir (int devPartIdx) {return  partitionPrefix + devPartIdx;}

    public String getPartitionAIModelPath (int devPartIdx) {return  partitionPrefix + devPartIdx + ProjectConstantPrefixes.AI_MODEL;}

    public String getPartitionACModelPath (int devPartIdx) {return  partitionPrefix + devPartIdx + ProjectConstantPrefixes.AC_MODEL;}

    public String getRerankerInstancesFilePath(int devPartIdx) {return partitionPrefix + devPartIdx + ProjectConstantPrefixes.RERANKER_INSTANCES_FILE;}

}

