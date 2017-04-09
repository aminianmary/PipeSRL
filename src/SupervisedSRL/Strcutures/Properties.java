package SupervisedSRL.Strcutures;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 9/13/16.
 */

public class Properties {
    private String trainFile;
    private String devFile;
    private String testFile;
    private String clusterFile;
    private String modelDir;
    private String outputDir;
    private int numOfPartitions;
    private int maxNumOfPITrainingIterations;
    private int maxNumOfPDTrainingIterations;
    private int maxNumOfAITrainingIterations;
    private int maxNumOfACTrainingIterations;
    private int maxNumOfRerankerTrainingIterations;
    private int numOfAIBeamSize;
    private int numOfACBeamSize;
    private int numOfPIFeatures;
    private int numOfPDFeatures;
    private int numOfAIFeatures;
    private int numOfACFeatures;
    private int numOfGlobalFeatures;
    private String indexMapFilePath;
    private String piModelPath;
    private String pdModelDir;
    private String aiModelPath;
    private String acModelPath;
    private String partitionPrefix;
    private String rerankerFeatureMapPath;
    private String rerankerSeenFeaturesPath;
    private String globalReverseLabelMapPath;
    private String rerankerModelPath;
    private String outputFilePathDev;
    private String outputFilePathTest;
    private String outputFilePathTest_w_projected_info;
    private boolean useReranker;
    private boolean usePI;
    private boolean supplementOriginalLabels;
    private String weightedLearning;
    private ArrayList<Integer> steps;
    private String modelsToBeTrained;
    private double aiCoefficient;
    private String trainPDLabelsPath;
    private String devPDLabelsPath;
    private String testPDLabelsPath;
    private String trainPILabelsPath;
    private String devPILabelsPath;
    private String testPILabelsPath;


    public Properties(String trainFile, String devFile, String testFile, String clusterFile, String modelDir, String outputDir,
                      int numOfPartitions, int maxNumOfPITrainingIterations, int maxNumOfPDTrainingIterations,int maxNumOfAITrainingIterations,
                      int maxNumOfACTrainingIterations, int maxNumOfRerankerTrainingIterations,
                      int numOfAIBeamSize, int numOfACBeamSize, int numOfPIFeatures, int numOfPDFeatures,
                      int numOfAIFeatures, int numOfACFeatures, int numOfGlobalFeatures,
                      boolean useReranker, String steps, String modelsToBeTrained, double aiCoefficient,
                      boolean pi, boolean supplementOriginalLabels, String weightedLearning) {
        this.numOfGlobalFeatures = numOfGlobalFeatures;
        this.trainFile = trainFile;
        this.devFile = devFile;
        this.testFile = testFile;
        this.clusterFile = clusterFile;
        this.modelDir = modelDir;
        this.outputDir = outputDir;
        this.numOfPartitions = numOfPartitions;
        this.maxNumOfPITrainingIterations = maxNumOfPITrainingIterations;
        this.maxNumOfPDTrainingIterations = maxNumOfPDTrainingIterations;
        this.maxNumOfAITrainingIterations = maxNumOfAITrainingIterations;
        this.maxNumOfACTrainingIterations = maxNumOfACTrainingIterations;
        this.maxNumOfRerankerTrainingIterations = maxNumOfRerankerTrainingIterations;
        this.numOfAIBeamSize = numOfAIBeamSize;
        this.numOfACBeamSize = numOfACBeamSize;
        this.numOfPIFeatures = numOfPIFeatures;
        this.numOfPDFeatures = numOfPDFeatures;
        this.numOfAIFeatures = numOfAIFeatures;
        this.numOfACFeatures = numOfACFeatures;
        this.indexMapFilePath = modelDir + ProjectConstants.INDEX_MAP;
        this.pdModelDir = modelDir;
        this.piModelPath = modelDir + ProjectConstants.PI_MODEL;
        this.aiModelPath = modelDir + ProjectConstants.AI_MODEL +".AIF_"+numOfAIFeatures;
        this.acModelPath = modelDir + ProjectConstants.AC_MODEL + ".ACF_"+numOfACFeatures;
        this.partitionPrefix = modelDir + ProjectConstants.PARTITION_PREFIX;
        this.rerankerFeatureMapPath = modelDir + ProjectConstants.RERANKER_FEATURE_MAP +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize+"."+aiCoefficient;
        this.rerankerSeenFeaturesPath = modelDir + ProjectConstants.RERANKER_SEEN_FEATURES +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize+"."+aiCoefficient;
        this.globalReverseLabelMapPath = acModelPath + ProjectConstants.GLOBAL_REVERSE_LABEL_MAP;
        this.rerankerModelPath = modelDir + ProjectConstants.RERANKER_MODEL +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize+"."+aiCoefficient;
        String reranker = (useReranker) ? "wr" : "wor";
        this.outputFilePathDev = outputDir + ProjectConstants.OUTPUT_FILE_DEV + "."+ reranker +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize+"."+aiCoefficient;
        this.outputFilePathTest = outputDir + ProjectConstants.OUTPUT_FILE_TEST + "."+ reranker +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize+"."+aiCoefficient;
        this.outputFilePathTest_w_projected_info = outputDir + ProjectConstants.OUTPUT_FILE_TEST + "."+ reranker +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize+"."+aiCoefficient+
                ProjectConstants.PROJECTED_INFO_SUFFIX;
        this.useReranker = useReranker;
        this.usePI = pi;
        this.supplementOriginalLabels = supplementOriginalLabels;
        this.weightedLearning = weightedLearning;
        this.aiCoefficient = aiCoefficient;
        this.steps = convertSteps2Array(steps);
        this.modelsToBeTrained = modelsToBeTrained;
        this.trainPDLabelsPath = modelDir + ProjectConstants.TRAIN_AUTO_PD_LABELS;
        this.devPDLabelsPath = modelDir + ProjectConstants.DEV_AUTO_PD_LABELS;
        this.testPDLabelsPath = modelDir + ProjectConstants.TEST_AUTO_PD_LABELS;
        this.trainPILabelsPath = modelDir + ProjectConstants.TRAIN_PI_LABELS;
        this.devPILabelsPath = modelDir + ProjectConstants.DEV_PI_LABELS;
        this.testPILabelsPath = modelDir + ProjectConstants.TEST_PI_LABELS;
        printModelProperties();
    }

    public String getTrainPDLabelsPath() {
        return trainPDLabelsPath;
    }

    public String getDevPDLabelsPath() {
        return devPDLabelsPath;
    }

    public String getTestPDLabelsPath() {
        return testPDLabelsPath;
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

    public String getTestFile() {
        return testFile;
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

    public int getMaxNumOfPITrainingIterations() {
        return maxNumOfPITrainingIterations;
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

    public String getOutputFilePathDev() {
        return outputFilePathDev;
    }

    public String getOutputFilePathTest() {
        return outputFilePathTest;
    }

    public String getOutputFilePathTest_w_projected_info() {
        return outputFilePathTest_w_projected_info;
    }

    public double getAiCoefficient() {
        return aiCoefficient;
    }

    public String getModelsToBeTrained() {
        return modelsToBeTrained;
    }

    public String getTrainPILabelsPath() {
        return trainPILabelsPath;
    }

    public String getDevPILabelsPath() {
        return devPILabelsPath;
    }

    public String getTestPILabelsPath() {
        return testPILabelsPath;
    }

    public String getPiModelPath() {
        return piModelPath;
    }

    public int getNumOfPIFeatures() {
        return numOfPIFeatures;
    }

    public String isWeightedLearning() {
        return weightedLearning;
    }

    public String getPartitionTrainDataPath(int devPartIdx) {
        return partitionPrefix + devPartIdx + ProjectConstants.PARTITION_TRAIN_DATA;
    }

    public String getPartitionDevDataPath(int devPartIdx) {
        return partitionPrefix + devPartIdx + ProjectConstants.PARTITION_DEV_DATA;
    }

    public String getPartitionDir(int devPartIdx) {
        return partitionPrefix + devPartIdx;
    }

    public String getPartitionPdModelDir(int devPartIdx) {
        return partitionPrefix + devPartIdx;
    }

    public String getPartitionPiModelPath(int devPartIdx) {
        return partitionPrefix + devPartIdx +  ProjectConstants.PI_MODEL;
    }

    public String getPartitionAIModelPath(int devPartIdx) {
        return partitionPrefix + devPartIdx + ProjectConstants.AI_MODEL +".AIF_"+numOfAIFeatures;
    }

    public String getPartitionACModelPath(int devPartIdx) {
        return partitionPrefix + devPartIdx + ProjectConstants.AC_MODEL + ".ACF_"+numOfACFeatures;
    }

    public String getRerankerInstancesFilePath(int devPartIdx) {
        return partitionPrefix + devPartIdx + ProjectConstants.RERANKER_INSTANCES_FILE +
                ".AIF_"+numOfAIFeatures + ".ACF_"+numOfACFeatures +".AIB_"+ numOfAIBeamSize +".ACB_"+ numOfACBeamSize;
    }

    public String getPartitionTrainPDAutoLabelsPath(int devPartIdx){
        return partitionPrefix + devPartIdx + ProjectConstants.TRAIN_AUTO_PD_LABELS;
    }

    public String getPartitionDevPDAutoLabelsPath(int devPartIdx){
        return partitionPrefix + devPartIdx + ProjectConstants.DEV_AUTO_PD_LABELS;
    }

    public String getPartitionTrainPILabelsPath(int devPartIdx){
        return partitionPrefix + devPartIdx + ProjectConstants.TRAIN_PI_LABELS;
    }

    public String getPartitionDevPILabelsPath(int devPartIdx){
        return partitionPrefix + devPartIdx + ProjectConstants.DEV_PI_LABELS;
    }

    public String getOutputDir() {
        return outputDir;
    }

    public boolean useReranker() {
        return useReranker;
    }

    public boolean usePI() {
        return usePI;
    }

    public boolean supplementOriginalLabels() {return supplementOriginalLabels;}

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
                "Test File Path: "+ testFile +"\n" +
                "Cluster File Path: "+ clusterFile+"\n" +
                "Model Directory: "+ modelDir+"\n" +
                "Output Directory: "+ outputDir+"\n" +
                "IndexMap File Path: "+ indexMapFilePath+"\n" +
                "PI Model path: "+ piModelPath+"\n" +
                "PD Models Dir: "+ pdModelDir+"\n" +
                "AI Model Path: "+ aiModelPath+"\n" +
                "AC Model Path: "+ acModelPath+"\n" +
                "Reranker FeatureMap Path: "+ rerankerFeatureMapPath+"\n" +
                "Global Reverse LabeMap Path: "+ globalReverseLabelMapPath+"\n" +
                "Reranker Model Path: "+ rerankerModelPath+"\n" +
                "Dev Output File Path: "+ outputFilePathDev+"\n" +
                "Test Output File Path: "+ outputFilePathTest+"\n" +
                "Number of Partitions: "+ numOfPartitions+"\n" +
                "AI Beam Size: "+ numOfAIBeamSize+"\n" +
                "AC Beam Size: "+ numOfACBeamSize+"\n" +
                "PI Feature Size: "+ numOfPIFeatures+"\n" +
                "PD Feature Size: "+ numOfPDFeatures+"\n" +
                "AI Feature Size: "+ numOfAIFeatures+"\n" +
                "AC Feature Size: "+ numOfACFeatures+"\n" +
                "Global Feature Size: "+ numOfGlobalFeatures+"\n" +
                "Max Number of PI Iterations: "+ maxNumOfPITrainingIterations+"\n" +
                "Max Number of PD Iterations: "+ maxNumOfPDTrainingIterations+"\n" +
                "Max Number of AI Iterations: "+ maxNumOfAITrainingIterations+"\n" +
                "Max Number of AC Iterations: "+ maxNumOfACTrainingIterations+"\n" +
                "AI Coefficient: "+ aiCoefficient + "\n" +
                "Models trained: "+ modelsToBeTrained+"\n"+
                "Pipeline Steps: "+ steps+"\n");
        if (usePI)
            System.out.print("PI USED\n");
        else
            System.out.print("PI NOT USED\n");
        if (supplementOriginalLabels)
            System.out.print("Projected Labels are supplemented with predictions\n");
        else
            System.out.print("Projected Labels are NOT supplemented with predictions\n");
        System.out.print("Weighted learning option: "+weightedLearning+"\n");
        if (useReranker)
            System.out.print("Reranker USED\n****************************\n");
        else
            System.out.print("Reranker NOT USED\n****************************\n");
    }
}
