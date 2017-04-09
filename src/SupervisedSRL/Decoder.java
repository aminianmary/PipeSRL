package SupervisedSRL;

import SentenceStruct.Sentence;
import SentenceStruct.simplePA;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.*;
import ml.AveragedPerceptron;
import util.IO;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by Maryam Aminian on 5/24/16.
 */
public class Decoder {

    AveragedPerceptron piClassifier; //predicate identification (binary classifier) --> NOTE: will be null if we don't use PI
    AveragedPerceptron aiClassifier; //argument identification (binary classifier)
    AveragedPerceptron acClassifier; //argument classification (multi-class classifier)

    public Decoder(AveragedPerceptron piClassifier, AveragedPerceptron classifier, String state) {
        this.piClassifier = piClassifier;
        if (state.equals("AI")) {
            this.aiClassifier = classifier;
        } else if (state.equals("AC") || state.equals("joint")) {
            this.acClassifier = classifier;
        }
    }

    public Decoder(AveragedPerceptron piClassifier, AveragedPerceptron aiClassifier, AveragedPerceptron acClassifier) {
        this.piClassifier = piClassifier;
        this.aiClassifier = aiClassifier;
        this.acClassifier = acClassifier;
    }

    ////////////////////////////////// DECODE ////////////////////////////////////////////////////////

    public void decode(IndexMap indexMap, ArrayList<String> devSentencesInCONLLFormat,
                       int aiMaxBeamSize, int acMaxBeamSize, int numOfPIFeatures, int numOfPDFeatures,
                       int numOfAIFeatures, int numOfACFeatures, String outputFile, double aiCoefficient,
                       String pdModelDir, boolean usePI, boolean supplement) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");
        System.out.println("Decoding started (on dev data)...");
        long startTime = System.currentTimeMillis();
        BufferedWriter outputWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "UTF-8"));
        BufferedWriter outputScoresWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile+".score"), "UTF-8"));
        BufferedWriter outputWSourceWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile+ProjectConstants.PROJECTED_INFO_SUFFIX),
                "UTF-8"));

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());

            String devSentence = devSentencesInCONLLFormat.get(d);
            Sentence sentence = new Sentence(devSentence, indexMap);

            TreeMap<Integer, simplePA> prediction = (TreeMap<Integer, simplePA>) predict(sentence, indexMap,
                    aiMaxBeamSize, acMaxBeamSize, numOfPIFeatures, numOfPDFeatures, numOfAIFeatures,
                    numOfACFeatures, false, aiCoefficient, pdModelDir, usePI);

            SRLOutput output = IO.generateCompleteOutputSentenceInCoNLLFormat(sentence, devSentence,prediction,supplement);
            outputWriter.write(output.getSentence());
            outputWSourceWriter.write(output.getSentence_w_projected_info());
            outputScoresWriter.write(d+"\t"+ output.getConfidenceScore() +"\n");
        }
        System.out.println(devSentencesInCONLLFormat.size());
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
        outputWriter.flush();
        outputWriter.close();
        outputWSourceWriter.flush();
        outputWSourceWriter.close();
        outputScoresWriter.flush();
        outputScoresWriter.close();
    }

    ////////////////////////////////// PREDICT ////////////////////////////////////////////////////////

    public HashMap<Integer, simplePA> predictAI(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                                                int numOfPIFeatures, int numOfPDFeatures, int numOfAIFeatures,
                                                String pdModelDir, boolean usePI)
            throws Exception {
        HashMap<Integer, simplePA> predictedPAs = new HashMap<Integer, simplePA>();
        int[] sentenceLemmas = sentence.getLemmas();
        String[] sentenceLemmas_str = sentence.getLemmas_str();
        ArrayList<Integer> goldPredicateIndices = sentence.getPredicatesIndices();

        for (int wordIdx = 0; wordIdx < sentence.getLength(); wordIdx++) {
            boolean isPredicate = false;
            if (usePI){
                //automatic predicate identification
                Object[] featureVector = FeatureExtractor.extractPIFeatures(wordIdx, sentence, numOfPIFeatures, indexMap);
                String piPrediction = piClassifier.predict(featureVector);
                if (piPrediction.equals("1"))
                    isPredicate = true;
            }else
            {
                //gold predicate indices
                if (goldPredicateIndices.contains(wordIdx))
                    isPredicate = true;
            }

            if (isPredicate) {
                //identified as a predicate
                int pIdx = wordIdx;
                int plem = sentenceLemmas[pIdx];
                String pLabel = "";

                Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, sentence, numOfPDFeatures, indexMap);
                File f1 = new File(pdModelDir + "/" + plem);
                if (f1.exists() && !f1.isDirectory()) {
                    //seen predicates
                    AveragedPerceptron classifier = AveragedPerceptron.loadModel(pdModelDir + "/" + plem);
                    pLabel = classifier.predict(pdfeats);
                } else {
                    if (plem != indexMap.unknownIdx) {
                        pLabel = indexMap.int2str(plem) + ".01"; //seen pLem
                    } else {
                        pLabel = sentenceLemmas_str[pIdx] + ".01"; //unseen pLem
                    }
                }

                //having pd label, set pSense in the sentence
                sentence.setPDAutoLabels4ThisPredicate(pIdx, pLabel);
                ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = new ArrayList();
                aiCandidates = getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfAIFeatures);
                HashMap<Integer, String> highestScorePrediction = getHighestScorePredication(aiCandidates);
                predictedPAs.put(pIdx, new simplePA(pLabel, highestScorePrediction));
            }
        }
        return predictedPAs;
    }

    public Object predict(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                          int acMaxBeamSize, int numOfPIFeatures, int numOfPDFeatures, int numOfAIFeatures, int numOfACFeatures,
                          boolean use4Reranker, double aiCoefficient, String pdModelDir,
                          boolean usePI) throws Exception {

        TreeMap<Integer, simplePA> predictedPAs = new TreeMap<Integer, simplePA>();
        TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates = new TreeMap<Integer, Prediction4Reranker>();
        int[] sentenceLemmas = sentence.getLemmas();
        String[] sentenceLemmas_str = sentence.getLemmas_str();
        ArrayList<Integer> goldPredicateIndices = sentence.getPredicatesIndices();
        String[] labelMap = acClassifier.getLabelMap();

        for (int wordIdx = 1; wordIdx < sentence.getLength(); wordIdx++) {
            boolean isPredicate = false;
            if (usePI) {
                Object[] featureVector = FeatureExtractor.extractPIFeatures(wordIdx, sentence, numOfPIFeatures, indexMap);
                String piPrediction = piClassifier.predict(featureVector);
                if (piPrediction.equals("1"))
                    isPredicate = true;
            }else{
                if(goldPredicateIndices.contains(wordIdx))
                    isPredicate = true;
            }

            if (isPredicate) {
                //identified as a predicate
                int pIdx = wordIdx;
                int plem = sentenceLemmas[pIdx];
                String pLabel = "";

                Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, sentence, numOfPDFeatures, indexMap);
                File f1 = new File(pdModelDir + "/" + plem);
                if (f1.exists() && !f1.isDirectory()) {
                    //seen predicates
                    AveragedPerceptron classifier = AveragedPerceptron.loadModel(pdModelDir + "/" + plem);
                    pLabel = classifier.predict(pdfeats);
                } else {
                    if (plem != indexMap.unknownIdx) {
                        pLabel = indexMap.int2str(plem) + ".01"; //seen pLem
                    } else {
                        pLabel = sentenceLemmas_str[pIdx] + ".01"; //unseen pLem
                    }
                }

                //having pd label, set pSense in the sentence
                sentence.setPDAutoLabels4ThisPredicate(pIdx, pLabel);
                ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfAIFeatures);
                ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = getBestACCandidates(sentence,
                        pIdx, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures, aiCoefficient);

                if (use4Reranker)
                    predictedAIACCandidates.put(pIdx, new Prediction4Reranker(pLabel, aiCandidates, acCandidates));
                else {
                    HashMap<Integer, String> highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates, labelMap);
                    predictedPAs.put(pIdx, new simplePA(pLabel, highestScorePrediction));
                }
            }
        }

        if (use4Reranker)
            return predictedAIACCandidates;
        else
            return predictedPAs;
    }

    ////////////////////////////////// GET BEST CANDIDATES ///////////////////////////////////////////////

    public ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidates
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
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates, int maxBeamSize, int numOfFeatures, double aiCoefficient) throws Exception {
        int numOfLabels = acClassifier.getLabelMap().length;
        ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> finalACCandidates = new ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>>();

        for (Pair<Double, ArrayList<Integer>> aiCandidate : aiCandidates) {
            //for each AI candidate generated by aiClassifier
            double aiScore = aiCoefficient * aiCandidate.first;
            ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
            currentBeam.add(new Pair<>(aiScore, new ArrayList<Integer>()));

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

    private HashMap<Integer, String> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates,
             String[] labelMap) {

        double highestScore = Double.NEGATIVE_INFINITY;
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
        HashMap<Integer, String> wordIndexLabelMap = new HashMap<Integer, String>();

        ArrayList<Integer> acResult = acCandidates.get(bestAIIndex).get(bestACIndex).second;
        ArrayList<Integer> aiResult = aiCandidates.get(bestAIIndex).second;
        assert acResult.size() == aiResult.size();

        for (int i = 0; i < acResult.size(); i++) {
            if (!labelMap[acResult.get(i)].equals("0"))
                wordIndexLabelMap.put(aiResult.get(i), labelMap[acResult.get(i)]);
        }
        return wordIndexLabelMap;
    }


    private HashMap<Integer, String> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates) {

        TreeSet<Pair<Double, ArrayList<Integer>>> sortedCandidates = new TreeSet<Pair<Double, ArrayList<Integer>>>(aiCandidates);
        Pair<Double, ArrayList<Integer>> highestScorePair = sortedCandidates.pollLast();

        //after finding highest score sequence in the list of candidates
        HashMap<Integer, String> wordIndexLabelMap = new HashMap<Integer, String>();
        ArrayList<Integer> highestScoreSeq = highestScorePair.second;

        for (int index : highestScoreSeq) {
            wordIndexLabelMap.put(index, "1");
        }

        return wordIndexLabelMap;
    }
}