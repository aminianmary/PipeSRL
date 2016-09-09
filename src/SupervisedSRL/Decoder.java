package SupervisedSRL;

import SentenceStruct.Sentence;
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
    public static void decode(Decoder decoder, IndexMap indexMap, String devDataPath, String[] labelMap,
                              int aiMaxBeamSize, int acMaxBeamSize,
                              int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures,
                              String modelDir, String outputFile) throws Exception {

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
            Sentence sentence = new Sentence(devSentence, indexMap);

            predictions[d] = (TreeMap<Integer, Prediction>) decoder.predict(sentence, indexMap, aiMaxBeamSize, acMaxBeamSize,
                    numOfAIFeatures, numOfACFeatures, numOfPDFeatures, modelDir, false);

            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentence));
        }
        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, labelMap, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
    }
    ////////////////////////////////// PREDICT ////////////////////////////////////////////////////////

    public HashMap<Integer, Prediction> predictAI(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                                                  int numOfFeatures, String pdModelDir, int numOfPDFeatures)
            throws Exception {
        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, pdModelDir, numOfPDFeatures);
        HashMap<Integer, Prediction> predictedPAs = new HashMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = new ArrayList();
            HashMap<Integer, Integer> highestScorePrediction = new HashMap<Integer, Integer>();

            aiCandidates = getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfFeatures);
            highestScorePrediction = getHighestScorePredication(aiCandidates);
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }


    public Object predict(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                          int acMaxBeamSize, int numOfAIFeatures, int numOfACFeatures,
                          int numOfPDFeatures, String pdModelDir, boolean use4Reranker) throws Exception {

        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, pdModelDir, numOfPDFeatures);
        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();
        TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates = new TreeMap<Integer, Prediction4Reranker>();
        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);
            HashMap<Integer, Integer> highestScorePrediction = new HashMap<Integer, Integer>();

            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfAIFeatures);
            ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = getBestACCandidates(sentence, pIdx, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures);

            if (use4Reranker)
                predictedAIACCandidates.put(pIdx, new Prediction4Reranker(pLabel, aiCandidates, acCandidates));
            else {
                highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates);
                predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
            }
        }

        if (use4Reranker)
            return predictedAIACCandidates;
        else
            return predictedPAs;
    }

    ////////////////////////////////// GET BEST CANDIDATES ///////////////////////////////////////////////

    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidates
            (Sentence sentence, int pIdx, IndexMap indexMap, int maxBeamSize, int numOfFeatures) throws Exception {
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

    private ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getBestACCandidates
            (Sentence sentence, int pIdx, IndexMap indexMap,
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates, int maxBeamSize, int numOfFeatures) throws Exception {
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
}
