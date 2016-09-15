package SupervisedSRL.Reranker;

import SentenceStruct.Argument;
import SentenceStruct.PA;
import SentenceStruct.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Pair;
import SupervisedSRL.Strcutures.RerankerFeatureMap;
import util.IO;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Maryam Aminian on 8/19/16.
 */
public class RerankerInstanceGenerator {
    int numOfPartitions;

    public RerankerInstanceGenerator(int numOfPartitions) {
        this.numOfPartitions = numOfPartitions;
    }

    public static HashMap<Integer, Integer> getArgLabelMap(Pair<Double, ArrayList<Integer>> aiCandid,
                                                           Pair<Double, ArrayList<Integer>> acCandid,
                                                           String[] localClassifierLabelMap, HashMap<String, Integer> globalReverseLabelMap) {
        HashMap<Integer, Integer> argLabelMap = new HashMap<Integer, Integer>();
        assert aiCandid.second.size() == acCandid.second.size();
        for (int i = 0; i < aiCandid.second.size(); i++) {
            int wordIdx = aiCandid.second.get(i);
            int label = globalReverseLabelMap.get(localClassifierLabelMap[acCandid.second.get(i)]);
            assert !argLabelMap.containsKey(wordIdx);
            argLabelMap.put(wordIdx, label);
        }
        return argLabelMap;
    }

    public static HashMap<Integer, Integer>[] extractFinalRerankerFeatures(int pIdx, String pLabel, Sentence sentence,
                                                                           Pair<Double, ArrayList<Integer>> aiCandid,
                                                                           Pair<Double, ArrayList<Integer>> acCandid,
                                                                           int numOfAIFeats, int numOfACFeats,
                                                                           IndexMap indexMap, String[] localCalssifierLabelMap,
                                                                           HashMap<String, Integer> globalReverseLabelMap,
                                                                           HashMap<Object, Integer>[] rerankerFeatureMap) throws Exception {
        HashMap<Integer, Integer> argMap = getArgLabelMap(aiCandid, acCandid, localCalssifierLabelMap, globalReverseLabelMap);
        int numOfGlobalFeatures = 1;
        HashMap<Integer, Integer>[] rerankerFeatureVector = new HashMap[numOfAIFeats + numOfACFeats + numOfGlobalFeatures];

        for (int wordIdx = 0; wordIdx < sentence.getWords().length; wordIdx++) {
            //for each word in the sentence
            int aiLabel = (argMap.containsKey(wordIdx)) ? 1 : 0;
            int acLabel = (argMap.containsKey(wordIdx)) ? argMap.get(wordIdx) : -1;
            Object[] aiFeats = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfAIFeats, indexMap, true, aiLabel);
            Object[] acFeats = (acLabel == -1) ? null : FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfACFeats, indexMap, true, acLabel);
            addToRerankerFeats(rerankerFeatureVector, aiFeats, 0, rerankerFeatureMap, false, numOfAIFeats);
            addToRerankerFeats(rerankerFeatureVector, acFeats, aiFeats.length, rerankerFeatureMap, false, numOfAIFeats);
        }
        Object[] globalFeats = FeatureExtractor.extractGlobalFeatures(pIdx, pLabel, aiCandid, acCandid, localCalssifierLabelMap);
        addToRerankerFeats(rerankerFeatureVector, globalFeats, numOfAIFeats + numOfACFeats, rerankerFeatureMap, true, numOfAIFeats);
        return rerankerFeatureVector;
    }

    // todo write test for this!
    private static void addToRerankerFeats(HashMap<Integer, Integer>[] rerankerFeatureVector, Object[] feats, int offset,
                                           HashMap<Object, Integer>[] rerankerFeatureMap, boolean isGlobalFeatures,
                                           int numOfAIFeatures) {
        if (feats == null) return;
        for (int i = 0; i < feats.length; i++) {
            if (rerankerFeatureVector[offset + i] == null)
                rerankerFeatureVector[offset + i] = new HashMap<>();

            int featureIndex = RerankerFeatureMap.unseenFeatureIndex;
            if (isGlobalFeatures && rerankerFeatureMap[numOfAIFeatures + i].containsKey(feats[i]))
                featureIndex = rerankerFeatureMap[numOfAIFeatures + i].get(feats[i]);
            else if (!isGlobalFeatures && rerankerFeatureMap[i].containsKey(feats[i]))
                featureIndex = rerankerFeatureMap[i].get(feats[i]);

            if (featureIndex != RerankerFeatureMap.unseenFeatureIndex) {
                if (!rerankerFeatureVector[offset + i].containsKey(featureIndex))
                    rerankerFeatureVector[offset + i].put(featureIndex, 1);
                else {
                    int oldFrequency = rerankerFeatureVector[offset + i].get(featureIndex);
                    rerankerFeatureVector[offset + i].put(featureIndex, oldFrequency + 1);
                }
            }
        }
    }

    public static HashMap<Integer, HashMap<Integer, Integer>> getGoldArgLabelMap(Sentence sentence, HashMap<String, Integer> globalReverseLabelMap) {
        HashMap<Integer, HashMap<Integer, Integer>> goldArgLabelMap = new HashMap<>();
        ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa : goldPAs) {
            int goldPIdx = pa.getPredicateIndex();
            ArrayList<Argument> goldArgs = pa.getArguments();
            HashMap<Integer, Integer> goldArgMap = new HashMap<Integer, Integer>();
            for (Argument arg : goldArgs)
                goldArgMap.put(arg.getIndex(), globalReverseLabelMap.get(arg.getType()));
            goldArgLabelMap.put(goldPIdx, goldArgMap);
        }
        return goldArgLabelMap;
    }

    public static HashMap<Integer, Integer>[] extractRerankerFeatures4GoldAssignment(int pIdx,
                                                                                     Sentence sentence,
                                                                                     HashMap<Integer, Integer> goldMap,
                                                                                     int numOfAIFeats, int numOfACFeats,
                                                                                     int numOfGlobalFeatures,
                                                                                     IndexMap indexMap,
                                                                                     HashMap<String, Integer>
                                                                                             globalReverseLabelMap,
                                                                                     HashMap<Object, Integer>[]
                                                                                             rerankerFeatureMap)
            throws Exception {
        HashMap<Integer, Integer>[] rerankerFeatureVector = new HashMap[numOfAIFeats + numOfACFeats + numOfGlobalFeatures];
        String[] globalLabelMap = getLabelMap(globalReverseLabelMap);

        for (int wordIdx = 0; wordIdx < sentence.getWords().length; wordIdx++) {
            int aiLabel = (goldMap.containsKey(wordIdx)) ? 1 : 0;
            int acLabel = (goldMap.containsKey(wordIdx)) ? goldMap.get(wordIdx) : -1;

            Object[] aiFeats = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfAIFeats, indexMap, true, aiLabel);
            Object[] acFeats = acLabel == -1 ? null : FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfACFeats, indexMap, true, acLabel);

            addToRerankerFeats(rerankerFeatureVector, aiFeats, 0, rerankerFeatureMap, false, numOfAIFeats);
            addToRerankerFeats(rerankerFeatureVector, acFeats, numOfAIFeats, rerankerFeatureMap, false, numOfAIFeats);
        }
        String pLabel = sentence.getPredicatesInfo().get(pIdx);
        ArrayList<Integer> aiAssignment = new ArrayList<>();
        ArrayList<Integer> acAssignment = new ArrayList<>();
        for (int arg : goldMap.keySet()) {
            aiAssignment.add(arg);
            acAssignment.add(goldMap.get(arg));
        }
        Object[] globalFeats = FeatureExtractor.extractGlobalFeatures(pIdx, pLabel, new Pair<>(1.0, aiAssignment),
                new Pair<>(1.0, acAssignment), globalLabelMap);
        addToRerankerFeats(rerankerFeatureVector, globalFeats, numOfAIFeats + numOfACFeats, rerankerFeatureMap, true, numOfAIFeats);
        return rerankerFeatureVector;
    }

    public static String[] getLabelMap(HashMap<String, Integer> globalReverseLabelMap) {
        String[] labelMap = new String[globalReverseLabelMap.size()];
        for (String label : globalReverseLabelMap.keySet()) {
            int labelIndex = globalReverseLabelMap.get(label);
            labelMap[labelIndex] = label;
        }
        return labelMap;
    }

    public ArrayList<String>[] getPartitions(String trainFilePath) throws IOException {
        ArrayList<String>[] partitions = new ArrayList[numOfPartitions];
        ArrayList<String> trainSentences = IO.readCoNLLFile(trainFilePath);
        //Collections.shuffle(sentencesInCoNLLFormat);
        int partitionSize = (int) Math.ceil((double) trainSentences.size() / numOfPartitions);
        int startIndex = 0;
        int endIndex = 0;
        for (int i = 0; i < numOfPartitions; i++) {
            endIndex = startIndex + partitionSize;
            ArrayList<String> partitionSentences = new ArrayList<>();
            if (endIndex < trainSentences.size())
                partitionSentences = new ArrayList<>(trainSentences.subList(startIndex, endIndex));
            else
                partitionSentences = new ArrayList<>(trainSentences.subList(startIndex, trainSentences.size()));

            partitions[i] = partitionSentences;
            startIndex = endIndex;
        }
        return partitions;
    }
}