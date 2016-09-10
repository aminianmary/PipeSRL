package SupervisedSRL.Strcutures;

import SentenceStruct.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Pipeline;
import SupervisedSRL.Reranker.RerankerInstanceGenerator;
import util.IO;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.zip.GZIPOutputStream;

/**
 * Created by monadiab on 9/9/16.
 */
public class RerankerFeatureMap {
    HashMap<Object, Integer>[] featureMap;
    HashSet<Object>[] seenFeatures;
    public static final String unseenSymbol = ";;??;;";

    public RerankerFeatureMap(int numOfFeatures){
        this.featureMap = new HashMap[numOfFeatures];
        this.seenFeatures= new HashSet[numOfFeatures];
    }

    public void updateSeenRerankerFeatures(Object[] features, int offset) {
        for (int dim = 0; dim < features.length; dim++) {
            seenFeatures[offset+ dim].add(features[dim]);
        }
    }


    public void updateSeenFeatures4ThisPredicate(int pIdx, String pLabel, Sentence sentence,
                                                 Pair<Double, ArrayList<Integer>> aiCandid,
                                                 Pair<Double, ArrayList<Integer>> acCandid,
                                                 int numOfAIFeats, int numOfACFeats, IndexMap indexMap,
                                                 String[] labelMap, HashMap<String, Integer> globalReverseLabelMap)
            throws Exception {
        HashMap<Integer, Integer> argMap= RerankerInstanceGenerator.getArgLabelMap(aiCandid, acCandid, labelMap, globalReverseLabelMap);

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
                                                                       IndexMap indexMap, String[] labelMap) throws Exception {

        HashMap<Integer, Integer> goldArgMap= RerankerInstanceGenerator.getGoldArgLabelMap(sentence).get(pIdx);

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
                new Pair<>(1.0D, acAssignment), labelMap);
        updateSeenRerankerFeatures(globalFeats, numOfAIFeats);
    }

    public void buildRerankerFeatureMap (){
        //constructing featureDic
        int numOfFeatures= seenFeatures.length;
        int featureIndex = 1;
        //for each feature slot
        for (int dim = 0; dim < numOfFeatures; dim++) {
            //adding seen feature indices
            for (Object feat : seenFeatures[dim]) {
                featureMap[dim].put(feat, featureIndex++);
            }
            //unseen feature index
            featureMap[dim].put(unseenSymbol, featureIndex++);
            assert !seenFeatures[dim].contains(unseenSymbol);
        }
    }


    public void save (String filePath) throws IOException{
        FileOutputStream fos = new FileOutputStream(filePath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);
        writer.writeObject(featureMap);
        writer.close();
    }

}
