package SupervisedSRL;

import SentenceStruct.Sentence;
import SupervisedSRL.Reranker.RerankerInstanceGenerator;
import SupervisedSRL.Reranker.RerankerInstanceItem;
import SupervisedSRL.Reranker.RerankerPool;
import SupervisedSRL.Strcutures.*;
import ml.AveragedPerceptron;
import util.IO;

import java.awt.image.AreaAveragingScaleFilter;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step9 {

    public static void generateRerankerInstances(Properties properties) throws Exception {
        if (!properties.getSteps().contains(8) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 8 -- Generate Reranker Train Instances\n>>>>>>>>>>>>>\n");
        int numOfPartitions = properties.getNumOfPartitions();
        RerankerInstanceGenerator rig = new RerankerInstanceGenerator(numOfPartitions);
        ArrayList<String>[] trainDataPartitions = rig.getPartitions(properties.getTrainFile());
        System.out.println("Loading RerankerFeatureMap...");
        HashMap<Object, Integer>[] rerankerFeatureMap = IO.load(properties.getRerankerFeatureMapPath());
        System.out.println("Done!\nLoading indexMap...");
        IndexMap indexMap = IO.load(properties.getIndexMapFilePath());
        System.out.println("Done!");
        HashMap<String, Integer> globalReverseLabelMap = IO.load(properties.getGlobalReverseLabelMapPath());
        int numOfAIBeamSize = properties.getNumOfAIBeamSize();
        int numOfACBeamSize = properties.getNumOfACBeamSize();
        int numOfPIFeatures = properties.getNumOfPIFeatures();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int numOfGlobalFeatures = properties.getNumOfGlobalFeatures();
        double aiCoefficient = properties.getAiCoefficient();
        String pdModelDir = properties.getPdModelDir();

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            System.out.println("PART "+ devPartIdx);
            String rerankerInstanceFilePath = properties.getRerankerInstancesFilePath(devPartIdx);
            String aiModelPath4Partition = properties.getPartitionAIModelPath(devPartIdx);
            String acModelPath4Partition = properties.getPartitionACModelPath(devPartIdx);
            String piModelPath4Partition = properties.getPartitionPiModelPath(devPartIdx);
            AveragedPerceptron aiClassifier = IO.load(aiModelPath4Partition);
            AveragedPerceptron acClassifier = IO.load(acModelPath4Partition);
            AveragedPerceptron piClassifier = IO.load(piModelPath4Partition);

            generateRerankerInstances4ThisPartition(piClassifier, aiClassifier, acClassifier, trainDataPartitions[devPartIdx],
                    rerankerFeatureMap, indexMap, globalReverseLabelMap, numOfAIBeamSize, numOfACBeamSize,
                    numOfPIFeatures, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures,
                    rerankerInstanceFilePath, aiCoefficient, pdModelDir);
        }
    }

    private static void generateRerankerInstances4ThisPartition(AveragedPerceptron piClassifier, AveragedPerceptron aiClassifier,
            AveragedPerceptron acClassifier, ArrayList<String> devSentences, HashMap<Object, Integer>[] rerankerFeatureMap,
                                                                IndexMap indexMap, HashMap<String, Integer> globalReverseLabelMap,
                                                                int aiBeamSize, int acBeamSize, int numOfPIFeatures,
                                                                int numOfPDFeatures, int numOfAIFeatures, int numOfACFeatures,
                                                                int numOfGlobalFeatures, String rerankerInstancesFilePath,
                                                                double aiCoefficient, String pdModelDir) throws Exception {
        Decoder decoder = new Decoder(piClassifier, aiClassifier, acClassifier);
        String[] localClassifierLabelMap = acClassifier.getLabelMap();
        FileOutputStream fos = new FileOutputStream(rerankerInstancesFilePath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutputStream writer = new ObjectOutputStream(gz);
        for (int d = 0; d < devSentences.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentences.size());
            Sentence devSentence = new Sentence(devSentences.get(d), indexMap);
            HashMap<Integer, HashMap<Integer, Integer>> goldMap = RerankerInstanceGenerator.getGoldArgLabelMap(devSentence, globalReverseLabelMap);

            TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates4thisSen =
                    (TreeMap<Integer, Prediction4Reranker>) decoder.predict(devSentence, indexMap, aiBeamSize, acBeamSize,
                            numOfPIFeatures, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, true, aiCoefficient, pdModelDir);

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
                                numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures, indexMap, localClassifierLabelMap, globalReverseLabelMap, rerankerFeatureMap), "0"), false);
                    }
                }
                //add gold assignment to the pool
                rerankerPool.addInstance(new RerankerInstanceItem(RerankerInstanceGenerator.extractRerankerFeatures4GoldAssignment(pIdx, devSentence, goldMap4ThisPredicate,
                        numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures, indexMap, globalReverseLabelMap, rerankerFeatureMap), "1"), true);
                writer.writeObject(rerankerPool);
                writer.flush();
                writer.reset();
            }
        }
        writer.close();
    }
}
