package SupervisedSRL.Strcutures;

import SentenceStruct.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Reranker.RerankerInstanceGenerator;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by monadiab on 9/9/16.
 */
public class RerankerFeatureMap implements Serializable {
    public static final int unseenFeatureIndex = 0;
    HashMap<Object, Integer>[] featureMap;
    HashSet<Object>[] seenFeatures;
    private int numOfSeenFeatures;

    public RerankerFeatureMap(int numOfFeatures) {
        this.featureMap = new HashMap[numOfFeatures];
        this.seenFeatures = new HashSet[numOfFeatures];
    }

    public void updateSeenRerankerFeatures(Object[] features, int offset) {
        if (features != null) {
            for (int dim = 0; dim < features.length; dim++) {
                if (seenFeatures[offset + dim] == null)
                    seenFeatures[offset + dim] = new HashSet<>();
                seenFeatures[offset + dim].add(features[dim]);
            }
        }
    }


    public void updateSeenFeatures4ThisPredicate(int pIdx, String pLabel, Sentence sentence,
                                                 Pair<Double, ArrayList<Integer>> aiCandid,
                                                 Pair<Double, ArrayList<Integer>> acCandid,
                                                 int numOfAIFeats, int numOfACFeats, IndexMap indexMap,
                                                 String[] labelMap, HashMap<String, Integer> globalReverseLabelMap)
            throws Exception {
        HashMap<Integer, Integer> argMap = RerankerInstanceGenerator.getArgLabelMap(aiCandid, acCandid, labelMap, globalReverseLabelMap);

        Object[] aiFeats;
        Object[] acFeats;
        Object[] globalFeats;

        for (int wordIdx = 0; wordIdx < sentence.getWords().length; wordIdx++) {
            int aiLabel = (argMap.containsKey(wordIdx)) ? 1 : 0;
            int acLabel = (argMap.containsKey(wordIdx)) ? argMap.get(wordIdx) : -1;
            aiFeats = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfAIFeats, indexMap, true, aiLabel);
            acFeats = (acLabel == -1) ? null : FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfACFeats, indexMap, true, acLabel);
            updateSeenRerankerFeatures(aiFeats, 0);
            updateSeenRerankerFeatures(acFeats, 0);
        }
        globalFeats = FeatureExtractor.extractGlobalFeatures(pIdx, pLabel, aiCandid, acCandid, labelMap);
        updateSeenRerankerFeatures(globalFeats, numOfAIFeats);
    }


    public void updateSeenFeatures4GoldInstance(int pIdx, Sentence sentence,
                                                int numOfAIFeats, int numOfACFeats,
                                                IndexMap indexMap, HashMap<String, Integer> globalReverseLabelMap) throws Exception {

        HashMap<Integer, Integer> goldArgMap = RerankerInstanceGenerator.getGoldArgLabelMap(sentence, globalReverseLabelMap).get(pIdx);
        String[] globalLabelMap = RerankerInstanceGenerator.getLabelMap(globalReverseLabelMap);

        for (int wordIdx = 0; wordIdx < sentence.getWords().length; wordIdx++) {
            //for each word in the sentence
            int aiLabel = (goldArgMap.containsKey(wordIdx)) ? 1 : 0;
            int acLabel = (goldArgMap.containsKey(wordIdx)) ? goldArgMap.get(wordIdx) : -1;
            Object[] aiFeats = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfAIFeats, indexMap, true, aiLabel);
            Object[] acFeats = acLabel == -1 ? null : FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfACFeats, indexMap, true, acLabel);
            updateSeenRerankerFeatures(aiFeats, 0);
            updateSeenRerankerFeatures(acFeats, 0);
        }

        String pLabel = sentence.getPredicatesInfo().get(pIdx);
        ArrayList<Integer> aiAssignment = new ArrayList<Integer>();
        ArrayList<Integer> acAssignment = new ArrayList<Integer>();
        for (int arg : goldArgMap.keySet()) {
            aiAssignment.add(arg);
            acAssignment.add(goldArgMap.get(arg));
        }
        Object[] globalFeats = FeatureExtractor.extractGlobalFeatures(pIdx, pLabel, new Pair<>(1.0D, aiAssignment),
                new Pair<>(1.0D, acAssignment), globalLabelMap);
        updateSeenRerankerFeatures(globalFeats, numOfAIFeats);
    }

    public void buildRerankerFeatureMap() {
        //constructing featureDic
        int numOfFeatures = seenFeatures.length;
        int featureIndex = 1;
        //for each feature slot
        for (int dim = 0; dim < numOfFeatures; dim++) {
            //adding seen feature indices
            for (Object feat : seenFeatures[dim]) {
                if (featureMap[dim] == null)
                    featureMap[dim] = new HashMap<>();

                featureMap[dim].put(feat, featureIndex++);
            }
        }

        this.numOfSeenFeatures = featureIndex;
    }

    public HashMap<Object, Integer>[] getFeatureMap() {
        return featureMap;
    }

    public int getNumOfSeenFeatures() {
        return numOfSeenFeatures;
    }
}
