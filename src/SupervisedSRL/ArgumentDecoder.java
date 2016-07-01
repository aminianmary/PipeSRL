package SupervisedSRL;

import Sentence.*;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.BeamElement;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.Pair;
import ml.AveragedPerceptron;
import util.IO;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by Maryam Aminian on 5/24/16.
 */
public class ArgumentDecoder {

    AveragedPerceptron aiClassifier; //argument identification (binary classifier)
    AveragedPerceptron acClassifier; //argument classification (multi-class classifier)
    int[][] aiConfusionMatrix = new int[2][2];
    HashMap<Integer, int[]> acConfusionMatrix = new HashMap<Integer, int[]>();

    public ArgumentDecoder(AveragedPerceptron classifier, String state) {

        if (state.equals("AI")) {
            aiConfusionMatrix[0][0] = 0;
            aiConfusionMatrix[0][1] = 0;
            aiConfusionMatrix[1][0] = 0;
            aiConfusionMatrix[1][1] = 0;
            this.aiClassifier = classifier;
        }else if (state.equals("joint"))
        {
            Set<String> acLabelSet = classifier.getReverseLabelMap().keySet();

            for (int k = 0; k < acLabelSet.size(); k++) {
                int[] acGoldLabels = new int[acLabelSet.size()];
                this.acConfusionMatrix.put(k, acGoldLabels);
            }
            this.acClassifier = classifier;
        }
    }


    public ArgumentDecoder(AveragedPerceptron aiClassifier, AveragedPerceptron acClassifier) {

        this.aiClassifier = aiClassifier;
        this.acClassifier = acClassifier;

        aiConfusionMatrix[0][0] = 0;
        aiConfusionMatrix[0][1] = 0;
        aiConfusionMatrix[1][0] = 0;
        aiConfusionMatrix[1][1] = 0;
        Set<String> acLabelSet = acClassifier.getReverseLabelMap().keySet();

        for (int k = 0; k < acLabelSet.size()+1; k++) {
            int[] acGoldLabels = new int[acLabelSet.size()+1];
            this.acConfusionMatrix.put(k, acGoldLabels);
        }
    }


    public static void decode (ArgumentDecoder argumentDecoder, IndexMap indexMap, String devData,
                        int aiMaxBeamSize, int acMaxBeamSize, int numOfFeatures) throws Exception
    {
        DecimalFormat format = new DecimalFormat("##.00");

        System.out.println("Decoding started (on dev data)...");
        long startTime  = System.currentTimeMillis();
        boolean decode= true;
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());

            Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap, decode);
            argumentDecoder.predict(sentence, indexMap, aiMaxBeamSize, acMaxBeamSize, numOfFeatures);
        }
        argumentDecoder.computePrecisionRecall("AC");
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

    }


    public static void decode (ArgumentDecoder argumentDecoder, IndexMap indexMap, String devData,
                               int maxBeamSize, int numOfFeatures) throws Exception
    {
        DecimalFormat format = new DecimalFormat("##.00");

        System.out.println("Decoding started (on dev data)...");
        long startTime  = System.currentTimeMillis();
        boolean decode= true;
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());

            Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap, decode);
            argumentDecoder.predictJoint(sentence, indexMap, maxBeamSize, numOfFeatures);
        }
        argumentDecoder.computePrecisionRecall("joint");
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

    }



    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidates
            (Sentence sentence, PA pa, IndexMap indexMap, int maxBeamSize, int numOfFeatures)

    {
        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));

        int[] sentenceWords = sentence.getWords();
        Predicate currentPr = pa.getPredicate();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            if (wordIdx == currentPr.getIndex())
                continue;

            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractFeatures(currentPr, wordIdx, sentence, "AI", numOfFeatures, indexMap);

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
    private HashMap<Integer, Integer> getHighestScoreAISeq(Sentence sentence, PA pa, IndexMap indexMap, int numOfFeatures)
    {
        int[] sentenceWords = sentence.getWords();
        Predicate currentPr = pa.getPredicate();
        HashMap<Integer, Integer> highestScoreAISeq = new HashMap<Integer, Integer>();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            if (wordIdx == currentPr.getIndex())
                continue;

            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractFeatures(currentPr, wordIdx, sentence, "AI", numOfFeatures, indexMap);
            double score1 = aiClassifier.score(featVector)[1];

            if (score1 >= 0) {
                highestScoreAISeq.put(wordIdx, 1);
            }
        }

        return highestScoreAISeq;
    }


    private ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getBestACCandidates
            (Sentence sentence, PA pa, IndexMap indexMap,
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             int maxBeamSize, int numOfFeatures)

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
                Object[] featVector = FeatureExtractor.extractFeatures(pa.getPredicate(), wordIdx, sentence, "AC", numOfFeatures, indexMap);
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


    //this function is used to test ai-ac combination
    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestCandidates
            (Sentence sentence, PA pa, IndexMap indexMap,
             int maxBeamSize, int numOfFeatures)

    {
        String[] labelMap = acClassifier.getLabelMap();

            ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
            currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));


            // Gradual building of the beam for the words identified as an argument by AI classifier
        for (int wordIdx = 1; wordIdx < sentence.getWords().length ; wordIdx++) {
            if (wordIdx == pa.getPredicate().getIndex())
                continue;

                // retrieve candidates for the current word
                Object[] featVector = FeatureExtractor.extractFeatures(pa.getPredicate(), wordIdx, sentence, "AC", numOfFeatures, indexMap);
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
        ArrayList<Integer> highestScoreAISeq= new ArrayList<Integer>();

        if (highestScoreSeqAIIndex != -1)
            highestScoreAISeq= aiCandidates.get(highestScoreSeqAIIndex).second;

        for (int k = 0; k < highestScoreAISeq.size(); k++) {
            int argIdx = highestScoreAISeq.get(k);
            wordIndexLabelMap.put(argIdx, highestScoreACSeq.get(k));
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



    //this function is used to test ai-ac modules combined
    private HashMap<Integer, Integer> getHighestScorePredicationJoint
            (ArrayList<Pair<Double, ArrayList<Integer>>> candidates, int pIndex) {

        TreeSet<Pair<Double, ArrayList<Integer>>> acCandidates4ThisSeq = new TreeSet<Pair<Double, ArrayList<Integer>>>(candidates);
        Pair<Double, ArrayList<Integer>> highestScorePair = acCandidates4ThisSeq.pollLast();

        //after finding highest score sequence in the list of candidates
        HashMap<Integer, Integer> wordIndexLabelMap = new HashMap<Integer, Integer>();
        ArrayList<Integer> highestScoreSeq = new ArrayList<Integer>();

        highestScoreSeq = highestScorePair.second;

        int realIndex = 1;
        for (int k = 0; k < highestScoreSeq.size(); k++) {
            if (realIndex == pIndex)
                realIndex++;
            wordIndexLabelMap.put(realIndex, highestScoreSeq.get(k));
            realIndex++;
        }

        return wordIndexLabelMap;
    }



    private void compareWithGold(PA pa, String state, HashMap<Integer, Integer> highestScorePrediction) {

        ArrayList<Argument> goldArgs = pa.getArguments();
        HashMap<Integer, String> goldArgMap = getGoldArgMap(goldArgs);
        Set<Integer> goldArgsIndices = goldArgMap.keySet();

        HashSet<Integer> exclusiveGoldArgIndices = new HashSet(goldArgsIndices);
        HashSet<Integer> commonGoldPredictedArgIndices = new HashSet(highestScorePrediction.keySet());
        HashSet<Integer> exclusivePredicatedArgIndices = new HashSet(highestScorePrediction.keySet());

        exclusivePredicatedArgIndices.removeAll(goldArgsIndices); //contains argument indices only identified by AI module
        commonGoldPredictedArgIndices.retainAll(goldArgsIndices);
        exclusiveGoldArgIndices.removeAll(highestScorePrediction.keySet());

        aiConfusionMatrix[1][1] += commonGoldPredictedArgIndices.size();
        aiConfusionMatrix[1][0] += exclusivePredicatedArgIndices.size();
        aiConfusionMatrix[0][1] += exclusiveGoldArgIndices.size();

        if (state.equals("AC")) {
            HashMap<String, Integer> reverseLabelMap = acClassifier.getReverseLabelMap();
            for (int predictedArgIdx : highestScorePrediction.keySet()) {
                int predictedLabel = highestScorePrediction.get(predictedArgIdx);
                if (goldArgMap.containsKey(predictedArgIdx)) {
                    //ai_tp --> (ac_tp/ac_fp)
                    int goldLabel = reverseLabelMap.get(goldArgMap.get(predictedArgIdx));
                    acConfusionMatrix.get(predictedLabel)[goldLabel]++;
                } else {
                    //ai_fp --> ac_fp
                    acConfusionMatrix.get(predictedLabel)[acConfusionMatrix.size() - 1]++;
                }
            }

            //update acConfusionMatrix for false negatives
            for (int goldArgIdx : goldArgMap.keySet()) {
                if (!highestScorePrediction.containsKey(goldArgIdx)) {
                    //ai_fn --> ac_fn
                    int goldLabel = reverseLabelMap.get(goldArgMap.get(goldArgIdx));
                    acConfusionMatrix.get(acConfusionMatrix.size() - 1)
                            [goldLabel]++;
                }
            }
        }

    }



    private void compareWithGoldJoint(PA pa, HashMap<Integer, Integer> highestScorePrediction) {

        ArrayList<Argument> goldArgs = pa.getArguments();
        HashMap<Integer, String> goldArgMap = getGoldArgMap(goldArgs);
        Set<Integer> goldArgsIndices = goldArgMap.keySet();

        HashSet<Integer> exclusiveGoldArgIndices = new HashSet(goldArgsIndices);
        HashSet<Integer> commonGoldPredictedArgIndices = getNonZeroArgs(highestScorePrediction);
        HashSet<Integer> exclusivePredicatedArgIndices = getNonZeroArgs(highestScorePrediction);

        exclusivePredicatedArgIndices.removeAll(goldArgsIndices); //contains argument indices only identified by AI module
        commonGoldPredictedArgIndices.retainAll(goldArgsIndices);
        exclusiveGoldArgIndices.removeAll(highestScorePrediction.keySet());

        aiConfusionMatrix[1][1] += commonGoldPredictedArgIndices.size();
        aiConfusionMatrix[1][0] += exclusivePredicatedArgIndices.size();
        aiConfusionMatrix[0][1] += exclusiveGoldArgIndices.size();

        HashMap<String, Integer> reverseLabelMap = acClassifier.getReverseLabelMap();
        for (int predictedArgIdx : highestScorePrediction.keySet()) {
            int predictedLabel = highestScorePrediction.get(predictedArgIdx);
            if (goldArgMap.containsKey(predictedArgIdx)) {
                //ai_tp --> (ac_tp/ac_fp)
                int goldLabel = reverseLabelMap.get(goldArgMap.get(predictedArgIdx));
                acConfusionMatrix.get(predictedLabel)[goldLabel]++;
            } else {
                //ai_fp --> ac_fp
                acConfusionMatrix.get(predictedLabel)[reverseLabelMap.get("0")]++;
            }
        }

        //update acConfusionMatrix for false negatives
        for (int goldArgIdx : goldArgMap.keySet()) {
            if (!highestScorePrediction.containsKey(goldArgIdx)) {
                //ai_fn --> ac_fn
                int goldLabel = reverseLabelMap.get(goldArgMap.get(goldArgIdx));
                acConfusionMatrix.get(reverseLabelMap.get("0"))
                        [goldLabel]++;
            }
        }
    }


    private HashSet<Integer> getNonZeroArgs (HashMap<Integer, Integer> prediction)
    {
        HashSet<Integer> nonZeroArgs = new HashSet();
        for (int key: prediction.keySet())
            if (prediction.get(key) != 21)
                nonZeroArgs.add(key);

        return nonZeroArgs;
    }


    private HashMap<Integer, String> getGoldArgMap(ArrayList<Argument> args) {
        HashMap<Integer, String> goldArgMap= new HashMap<Integer, String>();
        for (Argument arg : args)
            goldArgMap.put(arg.getIndex(), arg.getType());
        return goldArgMap;
    }



    public void computePrecisionRecall(String state) {
        DecimalFormat format = new DecimalFormat("##.00");
        //binary classification
        int aiTP = aiConfusionMatrix[1][1];
        int aiFP = aiConfusionMatrix[1][0];
        int aiFN = aiConfusionMatrix[0][1];
        int total_ai_predictions = aiTP + aiFP;

        System.out.println("Total AI prediction " + total_ai_predictions);
        System.out.println("AI Precision: " +format.format((double) aiTP / (aiTP + aiFP)));
        System.out.println("AI Recall: " + format.format((double) aiTP / (aiTP + aiFN)));
        System.out.println("*********************************************");

        if (state.equals("AC") || state.equals("joint")) {
            String[] labelMap = acClassifier.getLabelMap();
            int total_ac_predictions = 0;
            int total_tp=0;
            int total_gold=0;

            //multi-class classification
            for (int predicatedLabel : acConfusionMatrix.keySet()) {
                if (predicatedLabel != acConfusionMatrix.size() - 1) {
                    int tp = acConfusionMatrix.get(predicatedLabel)[predicatedLabel]; //element on the diagonal
                    total_tp += tp;

                    int total_prediction = 0;
                    for (int element : acConfusionMatrix.get(predicatedLabel))
                        total_prediction += element;

                    if (predicatedLabel != acConfusionMatrix.size() - 1)
                        total_ac_predictions += total_prediction;

                    int total_gold_4_this_label = 0;

                    for (int pLabel : acConfusionMatrix.keySet())
                        total_gold_4_this_label += acConfusionMatrix.get(pLabel)[predicatedLabel];

                    total_gold += total_gold_4_this_label;

                    double precision = 100. * (double) tp / total_prediction;
                    double recall = 100. * (double) tp / total_gold_4_this_label;
                    System.out.println("Precision of label " + labelMap[predicatedLabel] + ": " + format.format(precision));
                    System.out.println("Recall of label " + labelMap[predicatedLabel] + ": " + format.format(recall));
                }
            }

            System.out.println("*********************************************");
            System.out.println("Total AC prediction " + format.format(total_ac_predictions));
            System.out.println("Total number of tp: "+ format.format(total_tp));

            double micro_precision = 100. * (double) total_tp / total_ac_predictions;
            double micro_recall = 100. * (double) total_tp / total_gold;
            double FScore = (2 * micro_precision * micro_recall ) / (micro_precision+ micro_recall);

            System.out.println("Micro Precision: " + format.format(micro_precision));
            System.out.println("Micro Recall: " + format.format(micro_recall));
            System.out.println("Averaged F1-score: " + format.format(FScore));
        }

    }


    public void predictAI (Sentence sentence, IndexMap indexMap, int aiMaxBeamSize, int numOfFeatures) throws Exception {
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa : pas) {

            // get best k argument assignment candidates
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = getBestAICandidates(sentence, pa, indexMap, aiMaxBeamSize, numOfFeatures);
            HashMap<Integer, Integer> highestScorePrediction  = getHighestScorePredication(aiCandidates);

            //comparing with gold argument (assuming the predicate is given) --> updates two confusion matrices
            compareWithGold(pa, "AI", highestScorePrediction);
        }

    }


    public void predict(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize, int acMaxBeamSize, int numOfFeatures) throws Exception {
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa : pas) {

            // get best k argument assignment candidates
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = getBestAICandidates(sentence, pa, indexMap,
                    aiMaxBeamSize, numOfFeatures);

            // get best <=l argument label for each of these k assignments
            ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = getBestACCandidates(sentence, pa,
                    indexMap, aiCandidates, acMaxBeamSize, numOfFeatures);
            HashMap<Integer, Integer> highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates);

            HashMap<Integer, Integer> highestScorePrediction2 = getHighestScoreAISeq(sentence, pa,
                    indexMap, numOfFeatures);
            //comparing with gold argument (assuming the predicate is given) --> updates two confusion matrices
            compareWithGold(pa, "AC", highestScorePrediction);
        }
    }


    //this function is used to test ai-ac modules combination
    public void predictJoint(Sentence sentence, IndexMap indexMap, int maxBeamSize, int numOfFeatures) throws Exception {
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa : pas) {
            // get best k argument labels
            ArrayList<Pair<Double, ArrayList<Integer>>> candidates = getBestCandidates(sentence, pa, indexMap, maxBeamSize, numOfFeatures);
            HashMap<Integer, Integer> highestScorePrediction = getHighestScorePredicationJoint(candidates, pa.getPredicateIndex());
            //comparing with gold argument (assuming the predicate is given) --> updates two confusion matrices
            compareWithGoldJoint(pa, highestScorePrediction);
        }
    }

}
