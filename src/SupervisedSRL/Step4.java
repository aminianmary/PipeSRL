package SupervisedSRL;
import SentenceStruct.Sentence;
import SupervisedSRL.Reranker.RerankerInstanceItem;
import SupervisedSRL.Strcutures.*;

import java.util.*;

import ml.AveragedPerceptron;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step4 {

    public static void Step4(Pair<AveragedPerceptron, AveragedPerceptron>[] trainedClassifiers,
                             ArrayList<String>[] devPartitions, IndexMap indexMap, HashMap<String, Integer> globalReverseLabelMap,
                             int aiBeamSize, int acBeamSize, int numOfAIFeatures, int numOfACFeatures,
                             int numOfPDFeatures, int numOfGlobalFeatures, String pdModelDir, String featureMapPath) throws Exception{

        int numOfPartitions = devPartitions.length;
        RerankerFeatureMap rerankerFeatureMap = new RerankerFeatureMap(numOfAIFeatures + numOfGlobalFeatures);

        for (int devPart = 0; devPart < numOfPartitions ; devPart++){
            Decoder decoder = new Decoder(trainedClassifiers[devPart].first, trainedClassifiers[devPart].second);
            String[] localClassifierLabelMap = trainedClassifiers[devPart].second.getLabelMap();
            ArrayList<String> devSentences = devPartitions[devPart];

            for (int d = 0; d < devSentences.size(); d++) {
                if (d % 1000 == 0)
                    System.out.println(d + "/" + devSentences.size());

                Sentence devSentence = new Sentence(devSentences.get(d), indexMap);
                TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates4thisSen =
                        (TreeMap<Integer, Prediction4Reranker>) decoder.predict(devSentence, indexMap, aiBeamSize, acBeamSize,
                                numOfAIFeatures, numOfACFeatures, numOfPDFeatures, pdModelDir,true);

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
                            indexMap,localClassifierLabelMap );
                }
            }
        }
        rerankerFeatureMap.buildRerankerFeatureMap();
        rerankerFeatureMap.save(featureMapPath);
    }

}
