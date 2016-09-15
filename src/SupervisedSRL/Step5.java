package SupervisedSRL;

import SentenceStruct.Sentence;
import SupervisedSRL.Reranker.RerankerInstanceGenerator;
import SupervisedSRL.Reranker.RerankerInstanceItem;
import SupervisedSRL.Reranker.RerankerPool;
import SupervisedSRL.Strcutures.*;
import ml.AveragedPerceptron;

import java.io.FileOutputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step5 {

    public static void generateRerankerInstances(Properties properties) throws Exception {
        if (!properties.getSteps().contains(5) || !properties.useReranker())
            return;
        int numOfPartitions = properties.getNumOfPartitions();
        Pair<AveragedPerceptron, AveragedPerceptron>[] trainedClassifiersOnPartitions = Step4.loadTrainedClassifiersOnPartitions(properties);
        RerankerInstanceGenerator rig = new RerankerInstanceGenerator(numOfPartitions);
        ArrayList<String>[] trainDataPartitions = rig.getPartitions(properties.getTrainFile());
        HashMap<Object, Integer>[] rerankerFeatureMap = ModelInfo.loadFeatureMap(properties.getRerankerFeatureMapPath());
        IndexMap indexMap = ModelInfo.loadIndexMap(properties.getIndexMapFilePath());
        HashMap<String, Integer> globalReverseLabelMap = ModelInfo.loadReverseLabelMap(properties.getGlobalReverseLabelMapPath());
        int numOfAIBeamSize = properties.getNumOfAIBeamSize();
        int numOfACBeamSize = properties.getNumOfACBeamSize();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int numOfGlobalFeatures = properties.getNumOfGlobalFeatures();

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            String pdModelDir4Partition = properties.getPartitionPdModelDir(devPartIdx);
            String rerankerInstanceFilePath = properties.getRerankerInstancesFilePath(devPartIdx);

            generateRerankerInstances4ThisPartition(trainedClassifiersOnPartitions[devPartIdx], trainDataPartitions[devPartIdx],
                    rerankerFeatureMap, indexMap, globalReverseLabelMap, numOfAIBeamSize, numOfACBeamSize,
                    numOfAIFeatures, numOfACFeatures, numOfPDFeatures, numOfGlobalFeatures,
                    pdModelDir4Partition, rerankerInstanceFilePath);
        }
    }

    public static void generateRerankerInstances4ThisPartition(Pair<AveragedPerceptron, AveragedPerceptron> trainedClassifier, ArrayList<String> devSentences,
                                                               HashMap<Object, Integer>[] rerankerFeatureMap, IndexMap indexMap, HashMap<String, Integer> globalReverseLabelMap,
                                                               int aiBeamSize, int acBeamSize, int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures,
                                                               int numOfGlobalFeatures, String pdModelDir, String rerankerInstancesFilePath) throws Exception {
        Decoder decoder = new Decoder(trainedClassifier.first, trainedClassifier.second);
        String[] localClassifierLabelMap = trainedClassifier.second.getLabelMap();
        FileOutputStream fos = new FileOutputStream(rerankerInstancesFilePath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);

        for (int d = 0; d < devSentences.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentences.size());
            Sentence devSentence = new Sentence(devSentences.get(d), indexMap);
            HashMap<Integer, HashMap<Integer, Integer>> goldMap = RerankerInstanceGenerator.getGoldArgLabelMap(devSentence, globalReverseLabelMap);

            TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates4thisSen =
                    (TreeMap<Integer, Prediction4Reranker>) decoder.predict(devSentence, indexMap, aiBeamSize, acBeamSize,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures, pdModelDir, true);

            //creating the pool
            for (int pIdx : predictedAIACCandidates4thisSen.keySet()) {
                String pLabel = predictedAIACCandidates4thisSen.get(pIdx).getPredicateLabel();
                HashMap<Integer, Integer> goldMap4ThisPredicate = goldMap.get(pIdx);

                ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = predictedAIACCandidates4thisSen.get(pIdx).getAiCandidates();
                ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = predictedAIACCandidates4thisSen.get(pIdx).getAcCandidates();
                RerankerPool rerankerPool = new RerankerPool();

                for (int i = 0; i < aiCandidates.size(); i++) {
                    Pair<Double, ArrayList<Integer>> aiCandid = aiCandidates.get(i);
                    ArrayList<Pair<Double, ArrayList<Integer>>> acCandids4thisAiCandid = acCandidates.get(i);

                    for (int j = 0; j < acCandids4thisAiCandid.size(); j++) {
                        Pair<Double, ArrayList<Integer>> acCandid = acCandids4thisAiCandid.get(j);
                        rerankerPool.addInstance(new RerankerInstanceItem(RerankerInstanceGenerator.extractFinalRerankerFeatures(pIdx, pLabel, devSentence, aiCandid, acCandid,
                                numOfAIFeatures, numOfACFeatures, indexMap, localClassifierLabelMap, globalReverseLabelMap, rerankerFeatureMap), "0"), false);
                    }
                }
                //add gold assignment to the pool
                rerankerPool.addInstance(new RerankerInstanceItem(RerankerInstanceGenerator.extractRerankerFeatures4GoldAssignment(pIdx, devSentence, goldMap4ThisPredicate,
                        numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures, indexMap, globalReverseLabelMap, rerankerFeatureMap), "1"), true);
                writer.writeObject(rerankerPool);
            }
            writer.flush();
        }
        writer.close();
    }
}
