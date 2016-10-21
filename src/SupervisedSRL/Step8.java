package SupervisedSRL;

import SentenceStruct.Sentence;
import SupervisedSRL.Strcutures.*;
import ml.AveragedPerceptron;
import util.IO;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step8 {

    public static void buildRerankerFeatureMap(Properties properties) throws java.lang.Exception {

        if (!properties.getSteps().contains(8) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 8 -- Building Reranker FeatureMap\n>>>>>>>>>>>>>\n");
        IndexMap indexMap = IO.load(properties.getIndexMapFilePath());
        HashMap<String, Integer> globalReverseLabelMap = IO.load(properties.getGlobalReverseLabelMapPath());
        int numOfPartitions = properties.getNumOfPartitions();
        int aiBeamSize = properties.getNumOfAIBeamSize();
        int acBeamSize = properties.getNumOfACBeamSize();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int numOfGlobalFeatures = properties.getNumOfGlobalFeatures();
        double aiCoefficient = properties.getAiCoefficient();
        String rerankerFeatureMapFilePath = properties.getRerankerFeatureMapPath();
        String rerankerSeenFeaturesFilePath = properties.getNumOfRerankerSeenFeaturesPath();

        assert globalReverseLabelMap.size() != 0;
        RerankerFeatureMap rerankerFeatureMap = new RerankerFeatureMap(numOfAIFeatures + numOfGlobalFeatures);

        for (int devPart = 0; devPart < numOfPartitions; devPart++) {
            System.out.println("PART "+devPart);

            String aiModelPath4Partition = properties.getPartitionAIModelPath(devPart);
            String acModelPath4Partition = properties.getPartitionACModelPath(devPart);
            String devPDAutoLabelsPath = properties.getPartitionDevPDAutoLabelsPath(devPart);
            Pair<AveragedPerceptron, AveragedPerceptron> trainedClassifiersOnThisPartition = ModelInfo.loadTrainedModels(aiModelPath4Partition, acModelPath4Partition);
            HashMap<Integer, String>[] devPDAutoLabels = IO.load(devPDAutoLabelsPath);
            Decoder decoder = new Decoder(trainedClassifiersOnThisPartition.first, trainedClassifiersOnThisPartition.second);
            String[] localClassifierLabelMap = trainedClassifiersOnThisPartition.second.getLabelMap();
            ArrayList<String> devSentences = IO.load(properties.getPartitionDevDataPath(devPart));

            for (int d = 0; d < devSentences.size(); d++) {
                if (d % 1000 == 0)
                    System.out.println(d + "/" + devSentences.size());

                Sentence devSentence = new Sentence(devSentences.get(d), indexMap);
                devSentence.setPDAutoLabels(devPDAutoLabels[d]);
                TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates4thisSen =
                        (TreeMap<Integer, Prediction4Reranker>) decoder.predict(devSentence, indexMap, aiBeamSize, acBeamSize,
                                numOfAIFeatures, numOfACFeatures, true, aiCoefficient, devPDAutoLabels[d]);

                for (int pIdx : predictedAIACCandidates4thisSen.keySet()) {
                    String pLabel = predictedAIACCandidates4thisSen.get(pIdx).getPredicateLabel();
                    ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = predictedAIACCandidates4thisSen.get(pIdx).getAiCandidates();
                    ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = predictedAIACCandidates4thisSen.get(pIdx).getAcCandidates();

                    for (int i = 0; i < aiCandidates.size(); i++) {
                        Pair<Double, ArrayList<Integer>> aiCandid = aiCandidates.get(i);
                        ArrayList<Pair<Double, ArrayList<Integer>>> acCandids4thisAiCandid = acCandidates.get(i);

                        for (int j = 0; j < acCandids4thisAiCandid.size(); j++) {
                            Pair<Double, ArrayList<Integer>> acCandid = acCandids4thisAiCandid.get(j);
                            rerankerFeatureMap.updateSeenFeatures4ThisPredicate(pIdx, pLabel, devSentence, aiCandid, acCandid,
                                    numOfAIFeatures, numOfACFeatures, indexMap,
                                    localClassifierLabelMap, globalReverseLabelMap);
                        }
                    }
                    //add gold instance feature to the featureMap
                    rerankerFeatureMap.updateSeenFeatures4GoldInstance(pIdx, devSentence, numOfAIFeatures, numOfACFeatures,
                            indexMap, globalReverseLabelMap);
                }
            }
        }
        rerankerFeatureMap.buildRerankerFeatureMap();
        IO.write(rerankerFeatureMap.getFeatureMap(), rerankerFeatureMapFilePath);
        IO.write(rerankerFeatureMap.getNumOfSeenFeatures(), rerankerSeenFeaturesFilePath);
    }
}
