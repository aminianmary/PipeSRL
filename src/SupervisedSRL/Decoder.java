package SupervisedSRL;

import Sentence.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.*;
import ml.AveragedPerceptron;
import util.IO;

import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by Maryam Aminian on 5/24/16.
 */
public class Decoder {

    AveragedPerceptron aiClassifier; //argument identification (binary classifier)
    AveragedPerceptron acClassifier; //argument classification (multi-class classifier)

    public Decoder(AveragedPerceptron classifier, String state) {

        if (state.equals("AI")) {
            this.aiClassifier = classifier;
        } else if (state.equals("AC") || state.equals("joint")) {
            this.acClassifier = classifier;
        }
    }


    public Decoder(AveragedPerceptron aiClassifier, AveragedPerceptron acClassifier) {

        this.aiClassifier = aiClassifier;
        this.acClassifier = acClassifier;
    }


    ////////////////////////////////// DECODE ////////////////////////////////////////////////////////

    //stacked decoding
    public static void decode(Decoder decoder, IndexMap indexMap, ClusterMap clusterMap, String devDataPath, String[] labelMap,
                              int aiMaxBeamSize, int acMaxBeamSize,
                              int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures,
                              String modelDir, String outputFile,
                              HashMap<Object, Integer>[] aiFeatDict,
                              HashMap<Object, Integer>[] acFeatDict,
                              boolean greedy) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");

        System.out.println("Decoding started (on dev data)...");
        long startTime = System.currentTimeMillis();
        boolean decode = true;
        ArrayList<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devDataPath);
        TreeMap<Integer, Prediction>[] predictions = new TreeMap[devSentencesInCONLLFormat.size()];
        ArrayList<ArrayList<String>> sentencesToWriteOutputFile = new ArrayList<ArrayList<String>>();

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());

            String devSentence = devSentencesInCONLLFormat.get(d);
            Sentence sentence = new Sentence(devSentence, indexMap, clusterMap);

            predictions[d] = (TreeMap<Integer, Prediction>) decoder.predict(sentence, indexMap, aiMaxBeamSize, acMaxBeamSize,
                    numOfAIFeatures, numOfACFeatures, numOfPDFeatures, modelDir, aiFeatDict, acFeatDict, greedy, false);

            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentence));
        }
        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, labelMap, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
    }

    //joint decoding
    public static void decode(Decoder decoder, IndexMap indexMap, ClusterMap clusterMap, String devData,
                              String[] labelMap, int maxBeamSize, int numOfFeatures, int numOfPDFeatures,
                              String modelDir, String outputFile, HashMap<Object, Integer>[] featDict,
                              boolean greedy) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");

        System.out.println("Decoding started (on dev data)...");
        long startTime = System.currentTimeMillis();
        boolean decode = true;
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
        TreeMap<Integer, Prediction>[] predictions = new TreeMap[devSentencesInCONLLFormat.size()];
        ArrayList<ArrayList<String>> sentencesToWriteOutputFile = new ArrayList<ArrayList<String>>();

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());
            String devSentence = devSentencesInCONLLFormat.get(d);
            Sentence sentence = new Sentence(devSentence, indexMap, clusterMap);
            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentence));

            predictions[d] = decoder.predictJoint(sentence, indexMap, maxBeamSize, numOfFeatures, numOfPDFeatures, modelDir, featDict, greedy);
        }

        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, labelMap, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
    }


    ////////////////////////////////// PREDICT ////////////////////////////////////////////////////////

    public HashMap<Integer, Prediction> predictAI(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                                                  int numOfFeatures, String modelDir, int numOfPDFeatures,
                                                  HashMap<Object, Integer>[] featDict,
                                                  boolean greedy)
            throws Exception {
        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);
        HashMap<Integer, Prediction> predictedPAs = new HashMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = new ArrayList();
            ArrayList<Integer> aiCandidatesGreedy = new ArrayList<Integer>();
            HashMap<Integer, Integer> highestScorePrediction = new HashMap<Integer, Integer>();

            if (!greedy) {
                aiCandidates = getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfFeatures, featDict);
                highestScorePrediction = getHighestScorePredication(aiCandidates);

            } else {
                aiCandidatesGreedy = getBestAICandidatesGreedy(sentence, pIdx, indexMap, numOfFeatures, featDict);
                for (int idx = 0; idx < aiCandidatesGreedy.size(); idx++) {
                    int wordIdx = aiCandidatesGreedy.get(idx);
                    highestScorePrediction.put(wordIdx, 1);
                }
            }

            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }

    public Object predict(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                          int acMaxBeamSize, int numOfAIFeatures, int numOfACFeatures,
                          int numOfPDFeatures, String modelDir,
                          HashMap<Object, Integer>[] aiFeatDict,
                          HashMap<Object, Integer>[] acFeatDict,
                          boolean greedy, boolean use4Reranker) throws Exception {

        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);
        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();
        TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates = new TreeMap<Integer, Prediction4Reranker>();
        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = new ArrayList();
            ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = new ArrayList();

            ArrayList<Integer> aiCandidatesGreedy = new ArrayList<Integer>();
            ArrayList<Integer> acCandidatesGreedy = new ArrayList<Integer>();
            HashMap<Integer, Integer> highestScorePrediction = new HashMap<Integer, Integer>();


            if (!greedy) {
                aiCandidates = getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfAIFeatures, aiFeatDict);
                acCandidates = getBestACCandidates(sentence, pIdx, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures, acFeatDict);
                if (use4Reranker)
                    predictedAIACCandidates.put(pIdx, new Prediction4Reranker(pLabel, aiCandidates, acCandidates));
                else
                    highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates);
            } else {
                aiCandidatesGreedy = getBestAICandidatesGreedy(sentence, pIdx, indexMap, numOfAIFeatures, aiFeatDict);
                acCandidatesGreedy = getBestACCandidatesGreedy(sentence, pIdx, indexMap, aiCandidatesGreedy, numOfACFeatures, acFeatDict);
                if (use4Reranker)
                    predictedAIACCandidates.put(pIdx, new Prediction4Reranker(pLabel, aiCandidates, acCandidates));
                else {
                    for (int idx = 0; idx < acCandidatesGreedy.size(); idx++) {
                        int wordIdx = aiCandidatesGreedy.get(idx);
                        int label = acCandidatesGreedy.get(idx);
                        highestScorePrediction.put(wordIdx, label);
                    }
                }
            }
            if (!use4Reranker)
                predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }

        if (use4Reranker)
            return predictedAIACCandidates;
        else
            return predictedPAs;
    }


    //this function is used for joint ai-ac decoding
    public TreeMap<Integer, Prediction> predictJoint(Sentence sentence, IndexMap indexMap,
                                                     int maxBeamSize, int numOfFeatures, int numOfPDFeatures,
                                                     String modelDir, HashMap<Object, Integer>[] featDict,
                                                     boolean greedy) throws Exception {

        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);
        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> candidates = new ArrayList();
            int[] candidatesGreedy = new int[sentence.getWords().length];
            HashMap<Integer, Integer> highestScorePrediction = new HashMap<Integer, Integer>();

            if (!greedy) {
                candidates = getBestJointCandidates(sentence, pIdx, indexMap, maxBeamSize, numOfFeatures, featDict);
                highestScorePrediction = getHighestScorePredicationJoint(candidates, pIdx);
            } else {
                candidatesGreedy = getBestJointCandidatesGreedy(sentence, pIdx, indexMap, numOfFeatures, featDict);
                for (int idx = 0; idx < candidatesGreedy.length; idx++) {
                    highestScorePrediction.put(idx, candidatesGreedy[idx]);
                }
            }
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }


    ////////////////////////////////// GET BEST CANDIDATES ///////////////////////////////////////////////

    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidates
            (Sentence sentence, int pIdx, IndexMap indexMap, int maxBeamSize, int numOfFeatures,
             HashMap<Object, Integer>[] featDict) throws Exception {
        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));

        int[] sentenceWords = sentence.getWords();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap, false, 0);
            double score0 = Double.POSITIVE_INFINITY;
            double score1 = Double.NEGATIVE_INFINITY;

            double[] scores = aiClassifier.score(featVector);
            score0 = scores[0];
            score1 = scores[1];

            // build an intermediate beam
            TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

            for (int index = 0; index < currentBeam.size(); index++) {
                double currentScore = currentBeam.get(index).first;
                BeamElement be0 = new BeamElement(index, currentScore + score0, 0);
                BeamElement be1 = new BeamElement(index, currentScore + score1, 1);

                newBeamHeap.add(be0);
                if (newBeamHeap.size() > maxBeamSize)
                    newBeamHeap.pollFirst();

                newBeamHeap.add(be1);
                if (newBeamHeap.size() > maxBeamSize)
                    newBeamHeap.pollFirst();
            }

            ArrayList<Pair<Double, ArrayList<Integer>>> newBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>(maxBeamSize);
            for (BeamElement beamElement : newBeamHeap) {
                ArrayList<Integer> newArrayList = new ArrayList<Integer>();
                for (int b : currentBeam.get(beamElement.index).second)
                    newArrayList.add(b);
                if (beamElement.label == 1)
                    newArrayList.add(wordIdx);
                newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score, newArrayList));
            }

            // replace the old beam with the intermediate beam
            currentBeam = newBeam;
        }
        return currentBeam;
    }


    //getting highest score AI candidate (AP/LL/Adam) without Beam Search
    private ArrayList<Integer> getBestAICandidatesGreedy
    (Sentence sentence, int pIdx, IndexMap indexMap, int numOfFeatures, HashMap<Object, Integer>[] featDict) throws Exception {
        int[] sentenceWords = sentence.getWords();
        ArrayList<Integer> aiCandids = new ArrayList<Integer>();
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap, false, 0);
            double score1 = aiClassifier.score(featVector)[1];
            if (score1 >= 0)
                aiCandids.add(wordIdx);
        }
        return aiCandids;
    }


    private ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getBestACCandidates
            (Sentence sentence, int pIdx, IndexMap indexMap,
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates, int maxBeamSize, int numOfFeatures,
             HashMap<Object, Integer>[] featDict) throws Exception {

        int numOfLabels = acClassifier.getLabelMap().length;

        ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> finalACCandidates = new ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>>();

        for (Pair<Double, ArrayList<Integer>> aiCandidate : aiCandidates) {
            //for each AI candidate generated by aiClassifier
            double aiScore = aiCandidate.first;
            ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
            currentBeam.add(new Pair<Double, ArrayList<Integer>>(aiScore, new ArrayList<Integer>()));

            // Gradual building of the beam for the words identified as an argument by AI classifier
            for (int wordIdx : aiCandidate.second) {
                // retrieve candidates for the current word
                Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap, false, 0);
                double[] labelScores = acClassifier.score(featVector);

                // build an intermediate beam
                TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

                for (int index = 0; index < currentBeam.size(); index++) {
                    double currentScore = currentBeam.get(index).first;

                    for (int labelIdx = 0; labelIdx < numOfLabels; labelIdx++) {
                        newBeamHeap.add(new BeamElement(index, currentScore + labelScores[labelIdx], labelIdx));
                        if (newBeamHeap.size() > maxBeamSize)
                            newBeamHeap.pollFirst();
                    }
                }

                ArrayList<Pair<Double, ArrayList<Integer>>> newBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>(maxBeamSize);

                for (BeamElement beamElement : newBeamHeap) {
                    ArrayList<Integer> newArrayList = new ArrayList<Integer>();
                    for (int b : currentBeam.get(beamElement.index).second)
                        newArrayList.add(b);
                    newArrayList.add(beamElement.label);
                    newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score, newArrayList));
                }

                // replace the old beam with the intermediate beam
                currentBeam = newBeam;
            }

            //current beam for this ai candidates is built
            finalACCandidates.add(currentBeam);
        }

        return finalACCandidates;
    }


    //getting highest score AC candidate (AP/LL/Adam) without Beam Search
    private ArrayList<Integer> getBestACCandidatesGreedy
    (Sentence sentence, int pIdx, IndexMap indexMap, ArrayList<Integer> aiCandidates, int numOfFeatures,
     HashMap<Object, Integer>[] featDict) throws Exception {

        ArrayList<Integer> acCandids = new ArrayList<Integer>();
        for (int aiCandidIdx = 0; aiCandidIdx < aiCandidates.size(); aiCandidIdx++) {
            int wordIdx = aiCandidates.get(aiCandidIdx);
            Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap, false, 0);
            double[] labelScores = acClassifier.score(featVector);
            int predictedLabel = argmax(labelScores);
            acCandids.add(predictedLabel);
        }
        assert aiCandidates.size() == acCandids.size();
        return acCandids;
    }


    //this function is used for joint ai-ac decoding (AP/LL/Adam)
    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestJointCandidates
    (Sentence sentence, int pIdx, IndexMap indexMap,
     int maxBeamSize, int numOfFeatures, HashMap<Object, Integer>[] featDict) throws Exception {

        int numOfLabels = 0;
        numOfLabels = acClassifier.getLabelMap().length;

        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));


        // Gradual building of the beam for all words in the sentence
        for (int wordIdx = 1; wordIdx < sentence.getWords().length; wordIdx++) {
            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap, false, 0);
            double[] labelScores = acClassifier.score(featVector);

            // build an intermediate beam
            TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

            for (int index = 0; index < currentBeam.size(); index++) {
                double currentScore = currentBeam.get(index).first;

                for (int labelIdx = 0; labelIdx < numOfLabels; labelIdx++) {
                    newBeamHeap.add(new BeamElement(index, currentScore + labelScores[labelIdx], labelIdx));
                    if (newBeamHeap.size() > maxBeamSize)
                        newBeamHeap.pollFirst();
                }
            }

            ArrayList<Pair<Double, ArrayList<Integer>>> newBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>(maxBeamSize);

            for (BeamElement beamElement : newBeamHeap) {
                ArrayList<Integer> newArrayList = (ArrayList<Integer>) currentBeam.get(beamElement.index).second.clone();
                newArrayList.add(beamElement.label);
                newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score, newArrayList));
            }
            // replace the old beam with the intermediate beam
            currentBeam = newBeam;
        }

        return currentBeam;
    }


    private int[] getBestJointCandidatesGreedy
            (Sentence sentence, int pIdx, IndexMap indexMap,
             int numOfFeatures, HashMap<Object, Integer>[] featDict) throws Exception {

        int numOfLabels = acClassifier.getLabelMap().length;
        int[] predictedLabels = new int[sentence.getWords().length]; //for each word in the sentence, we have a label (either zero or non-zero)
        predictedLabels[0] = -1; //label for root element


        for (int wordIdx = 1; wordIdx < sentence.getWords().length; wordIdx++) {
            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap, false, 0);
            double[] labelScores = acClassifier.score(featVector);
            int prediction = argmax(labelScores);
            predictedLabels[wordIdx] = prediction;
        }
        return predictedLabels;
    }


    private HashMap<Integer, Integer> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates) {

        double highestScore = Double.NEGATIVE_INFINITY;
        ArrayList<Integer> highestScoreACSeq = new ArrayList<Integer>();
        int highestScoreSeqAIIndex = -1;
        int bestAIIndex = 0;
        int bestACIndex = 0;

        for (int aiIndex = 0; aiIndex < aiCandidates.size(); aiIndex++) {
            for (int acIndex = 0; acIndex < acCandidates.get(aiIndex).size(); acIndex++) {
                Pair<Double, ArrayList<Integer>> ar = acCandidates.get(aiIndex).get(acIndex);
                double score = ar.first;
                if (score > highestScore) {
                    highestScore = score;
                    bestAIIndex = aiIndex;
                    bestACIndex = acIndex;
                }
            }
        }

        //after finding highest score sequence in the list of AC candidates
        HashMap<Integer, Integer> wordIndexLabelMap = new HashMap<Integer, Integer>();

        ArrayList<Integer> acResult = acCandidates.get(bestAIIndex).get(bestACIndex).second;
        ArrayList<Integer> aiResult = aiCandidates.get(bestAIIndex).second;
        assert acResult.size() == aiResult.size();

        for (int i = 0; i < acResult.size(); i++)
            wordIndexLabelMap.put(aiResult.get(i), acResult.get(i));
        return wordIndexLabelMap;
    }


    private HashMap<Integer, Integer> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates) {

        TreeSet<Pair<Double, ArrayList<Integer>>> sortedCandidates = new TreeSet<Pair<Double, ArrayList<Integer>>>(aiCandidates);
        Pair<Double, ArrayList<Integer>> highestScorePair = sortedCandidates.pollLast();

        //after finding highest score sequence in the list of candidates
        HashMap<Integer, Integer> wordIndexLabelMap = new HashMap<Integer, Integer>();
        ArrayList<Integer> highestScoreSeq = highestScorePair.second;

        for (int index : highestScoreSeq) {
            wordIndexLabelMap.put(index, 1);
        }

        return wordIndexLabelMap;
    }


    //this function is used for joint ai-ac modules
    private HashMap<Integer, Integer> getHighestScorePredicationJoint
    (ArrayList<Pair<Double, ArrayList<Integer>>> candidates, int pIndex) {

        TreeSet<Pair<Double, ArrayList<Integer>>> acCandidates4ThisSeq = new TreeSet<Pair<Double, ArrayList<Integer>>>(candidates);
        Pair<Double, ArrayList<Integer>> highestScorePair = acCandidates4ThisSeq.pollLast();

        //after finding highest score sequence in the list of candidates
        HashMap<Integer, Integer> wordIndexLabelMap = new HashMap<Integer, Integer>();
        ArrayList<Integer> highestScoreSeq = highestScorePair.second;

        int realIndex = 1;
        for (int k = 0; k < highestScoreSeq.size(); k++) {
            if (realIndex == pIndex)
                realIndex++;
            wordIndexLabelMap.put(realIndex, highestScoreSeq.get(k));
            realIndex++;
        }

        return wordIndexLabelMap;
    }


    private int argmax(double[] scores) {
        int argmax = -1;
        double max = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > max) {
                argmax = i;
                max = scores[i];
            }
        }
        return argmax;
    }

}
