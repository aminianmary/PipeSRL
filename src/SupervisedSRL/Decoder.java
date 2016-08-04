package SupervisedSRL;

import Sentence.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.BeamElement;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Pair;
import SupervisedSRL.Strcutures.Prediction;
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


    public static void decode(Decoder decoder, IndexMap indexMap, String devDataPath, String[] labelMap,
                              int aiMaxBeamSize, int acMaxBeamSize, int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures, String modelDir, String outputFile) throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        System.out.println("Decoding started (on dev data)...");
        long startTime = System.currentTimeMillis();
        boolean decode = true;
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devDataPath);
        TreeMap<Integer, Prediction>[] predictions = new TreeMap[devSentencesInCONLLFormat.size()];
        ArrayList<ArrayList<String>> sentencesToWriteOutputFile = new ArrayList<ArrayList<String>>();

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());

            String devSentence = devSentencesInCONLLFormat.get(d);
            Sentence sentence = new Sentence(devSentence, indexMap, decode);
            //todo think about the correct way to show final predications
            predictions[d] = decoder.predict(sentence, indexMap, aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures, modelDir);
            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentence));

        }
        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, labelMap, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));

    }


    public static void decode(Decoder decoder, IndexMap indexMap, String devData,
                              String[] labelMap,
                              int maxBeamSize, int numOfFeatures, int numOfPDFeatures,
                              String modelDir,
                              String outputFile) throws Exception {
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
            Sentence sentence = new Sentence(devSentence, indexMap, decode);
            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentence));

            predictions[d] = decoder.predictJoint(sentence, indexMap, maxBeamSize, numOfFeatures, numOfPDFeatures, modelDir);
        }

        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, labelMap, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
    }

    private static TreeSet<Pair<Double, ArrayList<Integer>>> convertArrayListOfPairs2TreeSetOfPairs(ArrayList<Pair<Double, ArrayList<Integer>>> arrayOfPairs) {
        TreeSet<Pair<Double, ArrayList<Integer>>> treeSetOfPairs = new TreeSet<Pair<Double, ArrayList<Integer>>>();
        for (Pair<Double, ArrayList<Integer>> pair : arrayOfPairs) {
            treeSetOfPairs.add(pair);
        }
        return treeSetOfPairs;
    }

    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidates
            (Sentence sentence, int pIdx, String pLabel, IndexMap indexMap, int maxBeamSize, int numOfFeatures) throws Exception

    {
        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));

        int[] sentenceWords = sentence.getWords();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {

            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, pLabel, wordIdx, sentence, numOfFeatures, indexMap);

            double[] scores = aiClassifier.score(featVector);
            double score0 = scores[0];
            double score1 = scores[1];

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
                //todo check if it works properly
                ArrayList<Integer> newArrayList = (ArrayList<Integer>) currentBeam.get(beamElement.index).second.clone();
                if (beamElement.label == 1)
                    newArrayList.add(wordIdx);
                newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score, newArrayList));
            }

            // replace the old beam with the intermediate beam
            currentBeam = newBeam;
        }

        return currentBeam;
    }

    //getting highest score AI candidate without Beam Search
    private HashMap<Integer, Integer> getHighestScoreAISeq(Sentence sentence, int pIdx, String pLabel, IndexMap indexMap, int numOfFeatures) throws Exception {
        int[] sentenceWords = sentence.getWords();
        HashMap<Integer, Integer> highestScoreAISeq = new HashMap<Integer, Integer>();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            if (wordIdx == pIdx)
                continue;

            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, pLabel, wordIdx, sentence, numOfFeatures, indexMap);
            double score1 = aiClassifier.score(featVector)[1];

            if (score1 >= 0) {
                highestScoreAISeq.put(wordIdx, 1);
            }
        }

        return highestScoreAISeq;
    }

    private ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getBestACCandidates
            (Sentence sentence, int pIdx, String pLabel, IndexMap indexMap,
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             int maxBeamSize, int numOfFeatures) throws Exception

    {
        String[] labelMap = acClassifier.getLabelMap();
        ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> finalACCandidates =
                new ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>>();


        for (Pair<Double, ArrayList<Integer>> aiCandidate : aiCandidates) {
            //for each AI candidate generated by aiClassifier
            Double aiScore = aiCandidate.first;
            ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
            currentBeam.add(new Pair<Double, ArrayList<Integer>>(aiScore, new ArrayList<Integer>()));


            // Gradual building of the beam for the words identified as an argument by AI classifier
            for (int wordIdx : aiCandidate.second) {

                // retrieve candidates for the current word
                Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, pLabel, wordIdx, sentence, numOfFeatures, indexMap);
                double[] labelScores = acClassifier.score(featVector);

                // build an intermediate beam
                TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

                for (int index = 0; index < currentBeam.size(); index++) {
                    double currentScore = currentBeam.get(index).first;

                    for (int labelIdx = 0; labelIdx < labelMap.length; labelIdx++) {
                        newBeamHeap.add(new BeamElement(index, currentScore + labelScores[labelIdx], labelIdx));
                        if (newBeamHeap.size() > maxBeamSize)
                            newBeamHeap.pollFirst();
                    }
                }

                ArrayList<Pair<Double, ArrayList<Integer>>> newBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>(maxBeamSize);

                for (BeamElement beamElement : newBeamHeap) {
                    //todo check if it works properly
                    ArrayList<Integer> newArrayList = (ArrayList<Integer>) currentBeam.get(beamElement.index).second.clone();
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

    //this function is used for joint ai-ac decoding
    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestCandidates
    (Sentence sentence, int pIdx, String pLabel, IndexMap indexMap,
     int maxBeamSize, int numOfFeatures) throws Exception

    {
        String[] labelMap = acClassifier.getLabelMap();

        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));


        // Gradual building of the beam for all words in the sentence
        for (int wordIdx = 1; wordIdx < sentence.getWords().length; wordIdx++) {
            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, pLabel, wordIdx, sentence, numOfFeatures, indexMap);
            double[] labelScores = acClassifier.score(featVector);

            // build an intermediate beam
            TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

            for (int index = 0; index < currentBeam.size(); index++) {
                double currentScore = currentBeam.get(index).first;

                for (int labelIdx = 0; labelIdx < labelMap.length; labelIdx++) {
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

    private HashMap<Integer, Integer> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates) {

        double highestScore = Double.MIN_VALUE;
        ArrayList<Integer> highestScoreACSeq = new ArrayList<Integer>();
        int highestScoreSeqAIIndex = -1;

        for (int aiCandidateIndex = 0; aiCandidateIndex < aiCandidates.size(); aiCandidateIndex++) {

            TreeSet<Pair<Double, ArrayList<Integer>>> acCandidates4ThisSeq = new TreeSet<Pair<Double, ArrayList<Integer>>>(acCandidates.get(aiCandidateIndex));
            //TreeSet<Pair<Double, ArrayList<Integer>>> acCandidates4ThisSeq= convertArrayListOfPairs2TreeSetOfPairs(acCandidates.get(aiCandidateIndex));

            Pair<Double, ArrayList<Integer>> highestScorePair = acCandidates4ThisSeq.pollLast();
            if (highestScore < highestScorePair.first) {
                highestScore = highestScorePair.first;
                highestScoreACSeq = highestScorePair.second;
                highestScoreSeqAIIndex = aiCandidateIndex;
            }
        }

        //after finding highest score sequence in the list of AC candidates
        HashMap<Integer, Integer> wordIndexLabelMap = new HashMap<Integer, Integer>();
        ArrayList<Integer> highestScoreAISeq = new ArrayList<Integer>();

        if (highestScoreSeqAIIndex != -1)
            highestScoreAISeq = aiCandidates.get(highestScoreSeqAIIndex).second;

        for (int k = 0; k < highestScoreAISeq.size(); k++) {
            int argIdx = highestScoreAISeq.get(k);
            wordIndexLabelMap.put(argIdx, highestScoreACSeq.get(k));
        }

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


    public HashMap<Integer, Prediction> predictAI(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                                                  int numOfFeatures, String modelDir, int numOfPDFeatures)
            throws Exception {

        //Predicate disambiguation step
        //System.out.println("Disambiguating predicates of this sentence...");

        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);

        /*
        HashMap<Integer, String> predictedPredicates= new HashMap<Integer, String>();
        ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa: goldPAs)
            predictedPredicates.put(pa.getPredicateIndex(), pa.getPredicateLabel());
        */

        HashMap<Integer, Prediction> predictedPAs = new HashMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = getBestAICandidates(sentence, pIdx, pLabel, indexMap, aiMaxBeamSize, numOfFeatures);
            HashMap<Integer, Integer> highestScorePrediction = getHighestScorePredication(aiCandidates);
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }


    public HashMap<Integer, Prediction> predictAC(Sentence sentence, IndexMap indexMap,
                                                  int acMaxBeamSize, int aiMaxBeamSize, int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures, String modelDir) throws Exception {


        //Predicate disambiguation step
        //System.out.println("Disambiguating predicates of this sentence...");
        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);

        /*
        ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        */

        HashMap<Integer, Prediction> predictedPAs = new HashMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            String pLabel = predictedPredicates.get(pIdx);

            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates =
                    getBestAICandidates(sentence, pIdx, pLabel, indexMap, aiMaxBeamSize, numOfAIFeatures);

            // get best <=l argument label for each of these k assignments
            ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = getBestACCandidates(sentence,
                    pIdx, pLabel, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures);

            HashMap<Integer, Integer> highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates);

            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }


    public TreeMap<Integer, Prediction> predict(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                                                int acMaxBeamSize, int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures, String modelDir) throws Exception {

        //Predicate disambiguation step
        //System.out.println("Disambiguating predicates of this sentence...");
        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);
        /*
        /////////////////////////////
        HashMap<Integer, String> predictedPredicates= new HashMap<Integer, String>();
        ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        HashMap<Integer, ArrayList<Pair<Double, ArrayList<Integer>>>> goldArgs =
                new HashMap<Integer, ArrayList<Pair<Double, ArrayList<Integer>>>>();

        for (PA pa: goldPAs) {
            predictedPredicates.put(pa.getPredicateIndex(), pa.getPredicateLabel());
            ArrayList<Integer> args =  new ArrayList<Integer>();
            for (Argument argument: pa.getArguments())
                args.add(argument.getIndex());

            ArrayList<Pair<Double, ArrayList<Integer>>> goldList = new ArrayList<Pair<Double, ArrayList<Integer>>>();
            Pair<Double, ArrayList<Integer>> temp = new Pair<Double, ArrayList<Integer>>(1.0, args);
            goldList.add(temp);

            goldArgs.put(pa.getPredicateIndex(), goldList);
        }
        /////////////////////////////
        */

        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();

        //if (predictedPredicates.keySet().size()==0)
        //System.out.print("no predicate predicted...");

        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);

            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = getBestAICandidates(sentence,
                    pIdx, pLabel, indexMap,
                    aiMaxBeamSize, numOfAIFeatures);

            //gold arguments
            //ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = goldArgs.get(pIdx);

            // get best <=l argument label for each of these k assignments
            ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = getBestACCandidates(sentence,
                    pIdx, pLabel, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures);
            HashMap<Integer, Integer> highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates);

            HashMap<Integer, Integer> highestScorePrediction2 = getHighestScoreAISeq(sentence, pIdx, pLabel,
                    indexMap, numOfAIFeatures);

            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }


    //this function is used for joint ai-ac decoding
    public TreeMap<Integer, Prediction> predictJoint(Sentence sentence, IndexMap indexMap,
                                                     int maxBeamSize, int numOfFeatures, int numOfPDFeatures,
                                                     String modelDir) throws Exception {

        //Predicate disambiguation step
        //System.out.println("Disambiguating predicates of this sentence...");
        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);

        /*
        HashMap<Integer, String> predictedPredicates= new HashMap<Integer, String>();
        ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa: goldPAs)
            predictedPredicates.put(pa.getPredicateIndex(), pa.getPredicateLabel());
         */

        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> candidates = getBestCandidates(sentence, pIdx, pLabel, indexMap, maxBeamSize, numOfFeatures);
            HashMap<Integer, Integer> highestScorePrediction = getHighestScorePredicationJoint(candidates, pIdx);
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));

        }
        return predictedPAs;
    }

}
