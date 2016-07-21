package SupervisedSRL;

import Sentence.*;
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

    AveragedPerceptron aiNominalClassifier; //argument identification- Nominal (binary classifier)
    AveragedPerceptron aiVerbalClassifier; //argument identification- Verbal (binary classifier)
    AveragedPerceptron acNominalClassifier; //argument classification- Nominal (multi-class classifier)
    AveragedPerceptron acVerbalClassifier; //argument classification- Verbal (multi-class classifier)


    public Decoder(AveragedPerceptron aiNominalClassifier, AveragedPerceptron aiVerbalClassifier,
                   AveragedPerceptron acNominalClassifier, AveragedPerceptron acVerbalClassifier) {

        this.aiNominalClassifier = aiNominalClassifier;
        this.aiVerbalClassifier = aiVerbalClassifier;
        this.acNominalClassifier = acNominalClassifier;
        this.acVerbalClassifier = acVerbalClassifier;
    }


    public static void decode (Decoder decoder, IndexMap indexMap, String devDataPath,
                               int aiMaxBeamSize, int acMaxBeamSize, int numOfPDFeatures,
                               int numOfAINominalFeatures,int numOfAIVerbalFeatures,
                               int numOfACNominalFeatures,int numOfACVerbalFeatures, String modelDir, String outputFile) throws Exception
    {
        DecimalFormat format = new DecimalFormat("##.00");

        System.out.println("Decoding started (on dev data)...");
        long startTime  = System.currentTimeMillis();
        boolean decode= true;
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devDataPath);
        TreeMap<Integer, Prediction>[] predictions = new TreeMap[devSentencesInCONLLFormat.size()];
        ArrayList<ArrayList<String>> sentencesToWriteOutputFile = new ArrayList<ArrayList<String>>();

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());

            String devSentence = devSentencesInCONLLFormat.get(d);
            Sentence sentence = new Sentence(devSentence, indexMap, decode);
            predictions[d] = decoder.predict(sentence, indexMap, aiMaxBeamSize, acMaxBeamSize, numOfPDFeatures,
                    numOfAINominalFeatures, numOfAIVerbalFeatures, numOfACNominalFeatures, numOfACVerbalFeatures, modelDir);

            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentence));

        }
        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

    }


    public static void decode (Decoder decoder, IndexMap indexMap, String devData,
                               int maxBeamSize, int numOfFeatures, int numOfPDFeatures,
                               String modelDir,
                               String outputFile) throws Exception
    {
        DecimalFormat format = new DecimalFormat("##.00");

        System.out.println("Decoding started (on dev data)...");
        long startTime  = System.currentTimeMillis();
        boolean decode= true;
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
        TreeMap<Integer, Prediction>[] predictions = new TreeMap[devSentencesInCONLLFormat.size()];
        ArrayList<ArrayList<String>> sentencesToWriteOutputFile = new ArrayList<ArrayList<String>>();

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());
            String devSentence = devSentencesInCONLLFormat.get(d);
            Sentence sentence = new Sentence(devSentence, indexMap, decode);
            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentence));

            predictions[d]= decoder.predictJoint(sentence, indexMap, maxBeamSize, numOfPDFeatures, numOfFeatures, modelDir);
        }

        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

    }



    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidates
            (Sentence sentence, int pIdx, String pLabel, IndexMap indexMap, int maxBeamSize, int numOfFeatures, boolean isNominal)

    {
        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));

        int[] sentenceWords = sentence.getWords();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            if (wordIdx == pIdx)
                continue;

            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, wordIdx, sentence, "AI", numOfFeatures, indexMap);
            double[] scores = (isNominal)? aiNominalClassifier.score(featVector): aiVerbalClassifier.score(featVector);
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
    private HashMap<Integer, Integer> getHighestScoreAISeqWOBeamSearch(Sentence sentence, int pIdx, String pLabel,
                                                                       IndexMap indexMap, int numOfFeatures, boolean isNominal)
    {
        int[] sentenceWords = sentence.getWords();
        HashMap<Integer, Integer> highestScoreAISeq = new HashMap<Integer, Integer>();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            if (wordIdx == pIdx)
                continue;

            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractFeatures (pIdx, pLabel, wordIdx, sentence, "AI", numOfFeatures, indexMap);
            double score1 = (isNominal) ? aiNominalClassifier.score(featVector)[1] : aiVerbalClassifier.score(featVector)[1];

            if (score1 >= 0) {
                highestScoreAISeq.put(wordIdx, 1);
            }
        }

        return highestScoreAISeq;
    }


    private ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getBestACCandidates
            (Sentence sentence, int pIdx, String pLabel, IndexMap indexMap,
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             int maxBeamSize, int numOfFeatures, boolean isNominal)

    {
        AveragedPerceptron acClassifier = (isNominal) ? acNominalClassifier : acVerbalClassifier;
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
                Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, wordIdx, sentence, "AC", numOfFeatures, indexMap);
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

    //todo look into joint decoding later!
    //this function is used for ai-ac joint decoding
    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestCandidates
            (Sentence sentence, int pIdx, String pLabel, IndexMap indexMap,
             int maxBeamSize, int numOfFeatures, boolean isNominal)

    {
        AveragedPerceptron acClassifier = (isNominal) ? acNominalClassifier : acVerbalClassifier;
        String[] labelMap = acClassifier.getLabelMap();

            ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
            currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));


            // Gradual building of the beam for the words identified as an argument by AI classifier
        for (int wordIdx = 1; wordIdx < sentence.getWords().length ; wordIdx++) {
            if (wordIdx == pIdx)
                continue;

                // retrieve candidates for the current word
                Object[] featVector = FeatureExtractor.extractFeatures( pIdx, pLabel, wordIdx, sentence, "AC", numOfFeatures, indexMap);
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


    private static TreeSet<Pair<Double, ArrayList<Integer>>> convertArrayListOfPairs2TreeSetOfPairs (ArrayList<Pair<Double, ArrayList<Integer>>> arrayOfPairs)
    {
        TreeSet<Pair<Double, ArrayList<Integer>>> treeSetOfPairs= new TreeSet<Pair<Double, ArrayList<Integer>>>();
        for (Pair<Double, ArrayList<Integer>> pair: arrayOfPairs)
        {
            treeSetOfPairs.add(pair);
        }
        return treeSetOfPairs;
    }


    private HashMap<Integer, String> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates, boolean isNominal) {

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
        HashMap<Integer, String> wordIndexLabelMap = new HashMap<Integer, String>();
        ArrayList<Integer> highestScoreAISeq= new ArrayList<Integer>();
        String[] labelMap = (isNominal) ? acNominalClassifier.getLabelMap() : acVerbalClassifier.getLabelMap();

        if (highestScoreSeqAIIndex != -1)
            highestScoreAISeq= aiCandidates.get(highestScoreSeqAIIndex).second;

        for (int k = 0; k < highestScoreAISeq.size(); k++) {
            int argIdx = highestScoreAISeq.get(k);
            wordIndexLabelMap.put(argIdx, labelMap[highestScoreACSeq.get(k)]);
        }

        return wordIndexLabelMap;
    }


    private HashMap<Integer, Integer> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> candidates) {

        TreeSet<Pair<Double, ArrayList<Integer>>> sortedCandidates = new TreeSet<Pair<Double, ArrayList<Integer>>>(candidates);
        Pair<Double, ArrayList<Integer>> highestScorePair = sortedCandidates.pollLast();

        //after finding highest score sequence in the list of candidates
        HashMap<Integer, Integer> wordIndexLabelMap = new HashMap<Integer, Integer>();
        ArrayList<Integer> highestScoreSeq = new ArrayList<Integer>();

        highestScoreSeq = highestScorePair.second;

        for (int k = 0; k < highestScoreSeq.size(); k++) {
            wordIndexLabelMap.put(k, highestScoreSeq.get(k));
        }

        return wordIndexLabelMap;
    }

    //todo check is it's implemented properly
    //this function is used to test ai-ac modules combined
    private HashMap<Integer, String> getHighestScorePredicationJoint
            (ArrayList<Pair<Double, ArrayList<Integer>>> candidates, int pIndex, boolean isNominal) {

        TreeSet<Pair<Double, ArrayList<Integer>>> acCandidates4ThisSeq = new TreeSet<Pair<Double, ArrayList<Integer>>>(candidates);
        Pair<Double, ArrayList<Integer>> highestScorePair = acCandidates4ThisSeq.pollLast();

        //after finding highest score sequence in the list of candidates
        HashMap<Integer, String> wordIndexLabelMap = new HashMap<Integer, String>();
        ArrayList<Integer> highestScoreSeq = new ArrayList<Integer>();

        highestScoreSeq = highestScorePair.second;
        String[] labelMap = (isNominal) ? acNominalClassifier.getLabelMap() : acVerbalClassifier.getLabelMap();
        int realIndex = 1;
        for (int k = 0; k < highestScoreSeq.size(); k++) {
            if (realIndex == pIndex)
                realIndex++;
            wordIndexLabelMap.put(realIndex, labelMap[highestScoreSeq.get(k)]);
            realIndex++;
        }

        return wordIndexLabelMap;
    }


    //todo check this function when wanted to use dev evaluation later!
    public void predictAI (Sentence sentence, IndexMap indexMap, int aiMaxBeamSize, int numOfPDFeatures, int numOfAIFeatures, String modelDir)
            throws Exception
    {
        //Predicate disambiguation step
        System.out.println("Disambiguating predicates of this sentence...");
        HashMap<Integer, String> predictedPredicates =PD.predict(sentence,indexMap, numOfPDFeatures, modelDir);

        for (int pIdx :predictedPredicates.keySet()) {

            // get best k argument assignment candidates
            String pLabel =predictedPredicates.get(pIdx);
            boolean isNominal = isNominal(sentence.getPosTags()[pIdx], indexMap);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = getBestAICandidates(sentence, pIdx, pLabel,
                    indexMap, aiMaxBeamSize, numOfAIFeatures, isNominal);
            HashMap<Integer, Integer> highestScorePrediction  = getHighestScorePredication(aiCandidates);
        }

    }


    public TreeMap<Integer, Prediction> predict(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize, int acMaxBeamSize,
                                                int numOfPDFeatures,
                                                int numOfAINominalFeatures,int numOfAIVerbalFeatures,
                                                int numOfACNominalFeatures,  int numOfACVerbalFeatures,  String modelDir) throws Exception {

        //Predicate disambiguation step
        System.out.println("Disambiguating predicates of this sentence...");
        HashMap<Integer, String> predictedPredicates =PD.predict(sentence,indexMap, numOfPDFeatures,  modelDir);

        /*
        HashMap<Integer, String> predictedPredicates= new HashMap<Integer, String>();
        ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa: goldPAs)
            predictedPredicates.put(pa.getPredicateIndex(), pa.getPredicateLabel());
        */

        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();

        if (predictedPredicates.keySet().size()==0)
            System.out.print("no predicate predicted...");

        for (int pIdx :predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel =predictedPredicates.get(pIdx);
            boolean isNominal = isNominal(sentence.getPosTags()[pIdx], indexMap);
            int numOfAIFeatures = (isNominal) ? numOfAINominalFeatures : numOfAIVerbalFeatures;
            int numOfACFeatures = (isNominal) ? numOfACNominalFeatures : numOfACVerbalFeatures;

            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = getBestAICandidates(sentence,
                    pIdx, pLabel, indexMap,
                    aiMaxBeamSize, numOfAIFeatures, isNominal);

            // get best <=l argument label for each of these k assignments
            ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = getBestACCandidates(sentence,
                    pIdx, pLabel, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures, isNominal);

            HashMap<Integer, String> highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates, isNominal);

            HashMap<Integer, Integer> highestScorePrediction2 = getHighestScoreAISeqWOBeamSearch(sentence, pIdx,pLabel ,
                    indexMap, numOfAIFeatures, isNominal);

            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }

    //todo look into this function for joint decoding
    //this function is used to test ai-ac modules combination
    public TreeMap<Integer, Prediction> predictJoint(Sentence sentence, IndexMap indexMap,
                                                                    int maxBeamSize, int numOfPDFeatures, int numOfFeatures,
                                                                    String modelDir) throws Exception {

        //Predicate disambiguation step
        System.out.println("Disambiguating predicates of this sentence...");
        HashMap<Integer, String> predictedPredicates =PD.predict(sentence,indexMap, numOfPDFeatures, modelDir);

        /*
        HashMap<Integer, String> predictedPredicates= new HashMap<Integer, String>();
        ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa: goldPAs)
            predictedPredicates.put(pa.getPredicateIndex(), pa.getPredicateLabel());
         */

        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();

        for (int pIdx :predictedPredicates.keySet()) {
            String pLabel =predictedPredicates.get(pIdx);
            boolean isNominal = isNominal(sentence.getPosTags()[pIdx], indexMap);
            ArrayList<Pair<Double, ArrayList<Integer>>> candidates = getBestCandidates(sentence, pIdx, pLabel, indexMap, maxBeamSize, numOfFeatures, isNominal);
            HashMap<Integer, String> highestScorePrediction = getHighestScorePredicationJoint(candidates, pIdx, isNominal);
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));

        }
        return predictedPAs;
    }


    private static boolean isNominal (int ppos, IndexMap indexMap)
    {
        String[] int2StringMap = indexMap.getInt2stringMap();
        String pos = int2StringMap[ppos];
        if (pos.startsWith("VB"))
            return false;
        else
            return true;
    }
}
