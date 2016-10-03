package SupervisedSRL.Strcutures;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 9/13/16.
 */

public class Properties {
    private String trainFile;
    private String devFile;
    private String clusterFile;
    private String modelDir;
    private String outputDir;
    private int numOfPartitions;
    private int maxNumOfPDTrainingIterations;
    private int maxNumOfAITrainingIterations;
    private int maxNumOfACTrainingIterations;
    private int maxNumOfRerankerTrainingIterations;
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
    private String rerankerSeenFeaturesPath;
    private String globalReverseLabelMapPath;
    private String rerankerModelPath;
    private String outputFilePath;
    private boolean useReranker;
    private ArrayList<Integer> steps;
    private String modelsToBeTrained;
    private double aiCoefficient;
    private String trainAutoPDLabelsPath;
    private String devAutoPDLabelsPath;

    public Properties(String trainFile, String devFile, String clusterFile, String modelDir, String outputDir,
                      int numOfPartitions, int maxNumOfPDTrainingIterations,int maxNumOfAITrainingIterations,
                      int maxNumOfACTrainingIterations, int maxNumOfRerankerTrainingIterations,
                      int numOfAIBeamSize, int numOfACBeamSize, int numOfPDFeatures, int numOfAIFeatures, int numOfACFeatures, int numOfGlobalFeatures,
                      boolean useReranker, String steps, String modelsToBeTrained, double aiCoefficient) {
        this.numOfGlobalFeatures = numOfGlobalFeatures;
        this.trainFile = trainFile;
        this.devFile = devFile;
        this.clusterFile = clusterFile;
        this.modelDir = modelDir;
        this.outputDir = outputDir;
        this.numOfPartitions = numOfPartitions;
        this.maxNumOfPDTrainingIterations = maxNumOfPDTrainingIterations;
        this.maxNumOfAITrainingIterations = maxNumOfAITrainingIterations;
        this.maxNumOfACTrainingIterations = maxNumOfACTrainingIterations;
        this.maxNumOfRerankerTrainingIterations = maxNumOfRerankerTrainingIterations;
        this.numOfAIBeamSize = numOfAIBeamSize;
        this.numOfACBeamSize = numOfACBeamSize;
        this.numOfPDFeatures = numOfPDFeatures;
        this.numOfAIFeatures = numOfAIFeatures;
        this.numOfACFeatures = numOfACFeatures;
        this.indexMapFilePath = modelDir + ProjectConstantPrefixes.INDEX_MAP;
        this.pdModelDir = modelDir;
        this.aiModelPath = modelDir + ProjectConstantPrefixes.AI_MODEL +".AIF_"+numOfAIFeatures;
        this.acModelPath = modelDir + ProjectConstantPrefixes.AC_MODEL + ".ACF_"+numOfACFeatures;
        this.partitionPrefix = modelDir + ProjectConstantPrefixes.PARTITION_PREFIX;
        this.rerankerFeatureMapPath = modelDir + ProjectConstantPrefixes.RERANKER_FEATURE_MAP +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize+"."+aiCoefficient;
        this.rerankerSeenFeaturesPath = modelDir + ProjectConstantPrefixes.RERANKER_SEEN_FEATURES +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize+"."+aiCoefficient;
        this.globalReverseLabelMapPath = acModelPath + ProjectConstantPrefixes.GLOBAL_REVERSE_LABEL_MAP;
        this.rerankerModelPath = modelDir + ProjectConstantPrefixes.RERANKER_MODEL +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize+"."+aiCoefficient;
        String reranker = (useReranker) ? "wr" : "wor";
        this.outputFilePath = outputDir + ProjectConstantPrefixes.OUTPUT_FILE + "."+ reranker +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize+"."+aiCoefficient;
        this.useReranker = useReranker;
        this.aiCoefficient = aiCoefficient;
        this.steps = convertSteps2Array(steps);
        this.modelsToBeTrained = modelsToBeTrained;
        this.trainAutoPDLabelsPath = modelDir + ProjectConstantPrefixes.TRAIN_AUTO_PD_LABELS;
        this.devAutoPDLabelsPath = modelDir + ProjectConstantPrefixes.DEV_AUTO_PD_LABELS;
        printModelProperties();
    }

    public String getTrainAutoPDLabelsPath() {
        return trainAutoPDLabelsPath;
    }

    public String getDevAutoPDLabelsPath() {
        return devAutoPDLabelsPath;
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

    public int getMaxNumOfPDTrainingIterations() {
        return maxNumOfPDTrainingIterations;
    }

    public int getMaxNumOfAITrainingIterations() {
        return maxNumOfAITrainingIterations;
    }

    public int getMaxNumOfACTrainingIterations() {
        return maxNumOfACTrainingIterations;
    }

    public int getMaxNumOfRerankerTrainingIterations() {
        return maxNumOfRerankerTrainingIterations;
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

    public String getNumOfRerankerSeenFeaturesPath(){
        return rerankerSeenFeaturesPath;
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

    public double getAiCoefficient() {
        return aiCoefficient;
    }

    public String getModelsToBeTrained() {
        return modelsToBeTrained;
    }

    public String getPartitionTrainDataPath(int devPartIdx) {
        return partitionPrefix + devPartIdx + ProjectConstantPrefixes.PARTITION_TRAIN_DATA;
    }

    public String getPartitionDevDataPath(int devPartIdx) {
        return partitionPrefix + devPartIdx + ProjectConstantPrefixes.PARTITION_DEV_DATA;
    }

    public String getPartitionDir(int devPartIdx) {
        return partitionPrefix + devPartIdx;
    }

    public String getPartitionPdModelDir(int devPartIdx) {
        return partitionPrefix + devPartIdx;
    }

    public String getPartitionAIModelPath(int devPartIdx) {
        return partitionPrefix + devPartIdx + ProjectConstantPrefixes.AI_MODEL +".AIF_"+numOfAIFeatures;
    }

    public String getPartitionACModelPath(int devPartIdx) {
        return partitionPrefix + devPartIdx + ProjectConstantPrefixes.AC_MODEL + ".ACF_"+numOfACFeatures;
    }

    public String getRerankerInstancesFilePath(int devPartIdx) {
        return partitionPrefix + devPartIdx + ProjectConstantPrefixes.RERANKER_INSTANCES_FILE +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize;
    }

    public String getPartitionTrainPDAutoLabelsPath(int devPartIdx){
        return partitionPrefix + devPartIdx + ProjectConstantPrefixes.TRAIN_AUTO_PD_LABELS;
    }

    public String getPartitionDevPDAutoLabelsPath(int devPartIdx){
        return partitionPrefix + devPartIdx + ProjectConstantPrefixes.DEV_AUTO_PD_LABELS;
    }

    public String getOutputDir() {
        return outputDir;
    }

    public boolean useReranker() {
        return useReranker;
    }

    public ArrayList<Integer> getSteps() {
        return steps;
    }

    private ArrayList<Integer> convertSteps2Array(String steps) {
        ArrayList<Integer> s = new ArrayList<>();
        for (String part : steps.trim().split(","))
            s.add(Integer.parseInt(part.trim()));
        return s;
    }

    private void printModelProperties (){

        System.out.print("\n************** MODEL PROPERTIES **************\n" +
                "Train File Path : "+ trainFile +"\n" +
                "Dev File Path: " + devFile +"\n" +
                "Cluster File Path: "+ clusterFile+"\n" +
                "Model Directory: "+ modelDir+"\n" +
                "Output Directory: "+ outputDir+"\n" +
                "IndexMap File Path: "+ indexMapFilePath+"\n" +
                "Predicate Disambiguation Models Dir: "+ pdModelDir+"\n" +
                "AI Model Path: "+ aiModelPath+"\n" +
                "AC Model Path: "+ acModelPath+"\n" +
                "Reranker FeatureMap Path: "+ rerankerFeatureMapPath+"\n" +
                "Global Reverse LabeMap Path: "+ globalReverseLabelMapPath+"\n" +
                "Reranker Model Path: "+ rerankerModelPath+"\n" +
                "Output File Path: "+ outputFilePath+"\n" +
                "Number of Partitions: "+ numOfPartitions+"\n" +
                "AI Beam Size: "+ numOfAIBeamSize+"\n" +
                "AC Beam Size: "+ numOfACBeamSize+"\n" +
                "AI Feature Size: "+ numOfAIFeatures+"\n" +
                "AC Feature Size: "+ numOfACFeatures+"\n" +
                "PD Feature Size: "+ numOfPDFeatures+"\n" +
                "Global Feature Size: "+ numOfGlobalFeatures+"\n" +
                "Max Number of PD Iterations: "+ maxNumOfPDTrainingIterations+"\n" +
                "Max Number of AI Iterations: "+ maxNumOfAITrainingIterations+"\n" +
                "Max Number of AC Iterations: "+ maxNumOfACTrainingIterations+"\n" +
                "AI Coefficient: "+ aiCoefficient + "\n" +
                "Models trained: "+ modelsToBeTrained+"\n"+
        "Pipeline Steps: "+ steps+"\n");
        if (useReranker)
            System.out.print("Reranker USED\n****************************\n");
        else
            System.out.print("Reranker NOT USED\n****************************\n");

    }
}

