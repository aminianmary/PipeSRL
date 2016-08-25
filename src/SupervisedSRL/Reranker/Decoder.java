package SupervisedSRL.Reranker;

import SupervisedSRL.Strcutures.*;
import ml.AveragedPerceptron;
import util.IO;

import java.util.ArrayList;
import Sentence.Sentence;
import Sentence.PA;
import Sentence.Argument;

import java.util.HashMap;
import java.util.TreeMap;

/**
 * Created by Maryam Aminian on 8/25/16.
 */
public class Decoder {
    AveragedPerceptron aiClasssifier;
    AveragedPerceptron acClasssifier;
    AveragedPerceptron reranker;
    IndexMap indexMap;
    String pdModelDir;

    public Decoder(AveragedPerceptron aiClasssifier, AveragedPerceptron acClasssifier, AveragedPerceptron reranker,
                   IndexMap indexMap, String pdModelDir) {
        this.aiClasssifier = aiClasssifier;
        this.acClasssifier = acClasssifier;
        this.reranker = reranker;
        this.indexMap= indexMap;
        this.pdModelDir = pdModelDir;
    }

    private int predict(RerankerPool pool){
        return reranker.argmax(pool, true);
    }

    private void decode (String testData, int numOfPDFeatures, int numOfAIFeatures, int numOfACFeatures,
                         int aiMaxBeamSize, int acMaxBeamSize, String modelDir, boolean greedy, String outputFile) throws Exception{
        SupervisedSRL.Decoder decoder = new SupervisedSRL.Decoder(this.aiClasssifier, this.acClasssifier);
        ArrayList<String> testSentences = IO.readCoNLLFile(testData);
        ArrayList<ArrayList<String>> sentencesToWriteOutputFile = new ArrayList<ArrayList<String>>();
        TreeMap<Integer, Prediction>[] predictions = new TreeMap[testSentences.size()];

        for (int senIdx=0; senIdx < testSentences.size(); senIdx++) {
            Sentence testSentence = new Sentence(testSentences.get(senIdx), indexMap);
            HashMap<Integer, HashMap<Integer, Integer>> goldMap = getGoldArgLabelMap(testSentence, acClasssifier.getReverseLabelMap());
            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(testSentences.get(senIdx)));
            TreeMap<Integer, Prediction> predictions4ThisSentence= new TreeMap<Integer, Prediction>();

            TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates4thisSen =
                    (TreeMap<Integer, Prediction4Reranker>) decoder.predict(testSentence, indexMap, aiMaxBeamSize, acMaxBeamSize,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures, modelDir,
                            null, null, ClassifierType.AveragedPerceptron, greedy, true);

            //creating the pool and making prediction
            predictions4ThisSentence= obtainFinalPrediction4Sentence(numOfAIFeatures, numOfACFeatures, aiMaxBeamSize, testSentence, goldMap, predictedAIACCandidates4thisSen);
            predictions[senIdx]= predictions4ThisSentence;
        }
        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, acClasssifier.getLabelMap(), outputFile);
    }

    private TreeMap<Integer, Prediction> obtainFinalPrediction4Sentence(int numOfAIFeatures, int numOfACFeatures, int aiMaxBeamSize, Sentence testSentence, HashMap<Integer, HashMap<Integer, Integer>> goldMap, TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates4thisSen) throws Exception {
        TreeMap<Integer, Prediction> predictions4ThisSentence = new TreeMap<Integer, Prediction>();
        for (int pIdx : predictedAIACCandidates4thisSen.keySet()) {
            RerankerPool rerankerPool = new RerankerPool();
            String pLabel = predictedAIACCandidates4thisSen.get(pIdx).getPredicateLabel();
            //to keep scores/feats of different assignments
            HashMap<Integer, Integer> goldMap4ThisPredicate = goldMap.get(pIdx);

            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = predictedAIACCandidates4thisSen.get(pIdx).getAiCandidates();
            ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = predictedAIACCandidates4thisSen.get(pIdx).getAcCandidates();

            for (int i = 0; i < aiCandidates.size(); i++) {
                Pair<Double, ArrayList<Integer>> aiCandid = aiCandidates.get(i);
                ArrayList<Pair<Double, ArrayList<Integer>>> acCandids4thisAiCandid = acCandidates.get(i);

                for (int j = 0; j < acCandids4thisAiCandid.size(); j++) {
                    Pair<Double, ArrayList<Integer>> acCandid = acCandids4thisAiCandid.get(j);
                    rerankerPool.addInstance(new RerankerInstanceItem(RerankerInstanceGenerator.extractRerankerFeatures(pIdx, pLabel, testSentence, aiCandid, acCandid,
                            numOfAIFeatures, numOfACFeatures, indexMap, acClasssifier.getLabelMap()), "0"), false);
                }
            }
            int bestCandidateIndex = predict(rerankerPool);
            int bestAICandidIndex = bestCandidateIndex/aiMaxBeamSize;
            int bestACCandidIndex = (bestAICandidIndex % aiMaxBeamSize) -1;
            predictions4ThisSentence.put(pIdx,
                    new Prediction(pLabel, aiCandidates.get(bestAICandidIndex).second ,
                            acCandidates.get(bestAICandidIndex).get(bestACCandidIndex).second));
        }
        return predictions4ThisSentence;
    }

    private HashMap<Integer, HashMap<Integer, Integer>> getGoldArgLabelMap (Sentence sentence,
                                                                            HashMap<String, Integer> reverseLabelMap){
        HashMap<Integer, HashMap<Integer, Integer>> goldArgLabelMap = new HashMap<Integer, HashMap<Integer, Integer>>();
        ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa: goldPAs){
            int goldPIdx = pa.getPredicateIndex();
            ArrayList<Argument> goldArgs=  pa.getArguments();
            HashMap<Integer, Integer> goldArgMap = new HashMap<Integer, Integer>();
            for (Argument arg: goldArgs)
                goldArgMap.put(arg.getIndex(), reverseLabelMap.get(arg.getType()));
            goldArgLabelMap.put(goldPIdx, goldArgMap);
        }
        return goldArgLabelMap;
    }

}
