package SupervisedSRL;

import Sentence.*;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.BeamElement;
import SupervisedSRL.Strcutures.Pair;
import ml.AveragedPerceptron;

import java.net.Inet4Address;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Created by Maryam Aminian on 5/24/16.
 */
public class ArgumentDecoder {

    AveragedPerceptron aiClassifier; //argument identification (binary classifier)
    AveragedPerceptron acClassifier; //argument classification (multi-class classifier)
    int[][] aiConfusionMatrix = new int[2][2];
    HashMap<Integer, int[]> acConfusionMatrix = new HashMap<Integer, int[]>();

    public ArgumentDecoder(AveragedPerceptron aiClassifier, AveragedPerceptron acClassifier, HashSet<String> acLabelSet) {

        aiConfusionMatrix[0][0] = 0;
        aiConfusionMatrix[0][1] = 0;
        aiConfusionMatrix[1][0] = 0;
        aiConfusionMatrix[1][1] = 0;

        for (int k = 0; k < acLabelSet.size() + 1; k++) {
            int[] acGoldLabels = new int[acLabelSet.size() + 1];
            this.acConfusionMatrix.put(k, acGoldLabels);
        }

        this.aiClassifier = aiClassifier;
        this.acClassifier = acClassifier;
    }


    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidates
            (Sentence sentence, PA pa, int maxBeamSize)

    {
        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));

        String[] sentenceWords = sentence.getWords();
        Predicate currentPr = pa.getPredicate();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            if (wordIdx == currentPr.getIndex())
                continue;

            // retrieve candidates for the current word
            String[] featVector = FeatureExtractor.extractFeatures(currentPr, wordIdx, sentence, "AI", 93);
            List<String> features = Arrays.asList(featVector);
            double score0 = aiClassifier.score(features, "0");
            double score1 = aiClassifier.score(features, "1");

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
    private HashMap<Integer, Integer> getHighestScoreAISeq(Sentence sentence, PA pa)
    {
        String[] sentenceWords = sentence.getWords();
        Predicate currentPr = pa.getPredicate();
        HashMap<Integer, Integer> highestScoreAISeq = new HashMap<Integer, Integer>();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            if (wordIdx == currentPr.getIndex())
                continue;
            Pipeline.testSize++;

            // retrieve candidates for the current word
            String[] featVector = FeatureExtractor.extractFeatures(currentPr, wordIdx, sentence, "AI", 93);
            List<String> features = Arrays.asList(featVector);
            double score1 = aiClassifier.score(features, "1");

            if (score1 >= 0) {
                highestScoreAISeq.put(wordIdx, 1);
            }
        }

        return highestScoreAISeq;
    }


    private ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getBestACCandidates
            (Sentence sentence,
             PA pa,
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             int maxBeamSize)

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
                String[] featVector = FeatureExtractor.extractFeatures(pa.getPredicate(), wordIdx, sentence, "AC", 93);
                List<String> features = Arrays.asList(featVector);

                Double[] labelScores = acClassifier.score(features);

                // build an intermediate beam
                TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

                for (int index = 0; index < currentBeam.size(); index++) {
                    double currentScore = currentBeam.get(index).first;

                    BeamElement[] bes = new BeamElement[labelMap.length];

                    for (int labelIdx = 0; labelIdx < labelMap.length; labelIdx++) {
                        bes[labelIdx] = new BeamElement(index, currentScore + labelScores[labelIdx], labelIdx);
                        newBeamHeap.add(bes[labelIdx]);
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


    private HashMap<Integer, Integer> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates) {
        double highestScore = Double.MIN_VALUE;
        ArrayList<Integer> highestScoreACSeq = new ArrayList<Integer>();
        int highestScoreSeqAIIndex = -1;

        for (int aiCandidateIndex = 0; aiCandidateIndex < aiCandidates.size(); aiCandidateIndex++) {
            TreeSet<Pair<Double, ArrayList<Integer>>> acCandidates4ThisSeq = new TreeSet<Pair<Double, ArrayList<Integer>>>(acCandidates.get(aiCandidateIndex));
            Pair<Double, ArrayList<Integer>> highestScorePair = acCandidates4ThisSeq.pollLast();
            if (highestScore < highestScorePair.first) {
                highestScore = highestScorePair.first;
                highestScoreACSeq = highestScorePair.second;
                highestScoreSeqAIIndex = aiCandidateIndex;
            }
        }

        //after finding highest score sequence in the list of AC candidates
        HashMap<Integer, Integer> wordIndexLabelMap = new HashMap<Integer, Integer>();
        ArrayList<Integer> highestScoreAISeq = aiCandidates.get(highestScoreSeqAIIndex).second;

        for (int k = 0; k < highestScoreAISeq.size(); k++) {
            int argIdx = highestScoreAISeq.get(k);
            wordIndexLabelMap.put(argIdx, highestScoreACSeq.get(k));
        }

        return wordIndexLabelMap;
    }


    private void compareWithGold(PA pa, HashMap<Integer, Integer> highestScorePrediction) {

        HashMap<String, Integer> reverseLabelMap = acClassifier.getReverseLabelMap();

        ArrayList<Argument> goldArgs = pa.getArguments();
        HashSet<Integer> goldArgsIndices = getArgIndices(goldArgs);

        HashSet<Integer> exclusiveGoldArgIndices = new HashSet(goldArgsIndices);
        HashSet<Integer> commonGoldPredictedArgIndices = new HashSet(highestScorePrediction.keySet());
        HashSet<Integer> exclusivePredicatedArgIndices = new HashSet(highestScorePrediction.keySet());

        exclusivePredicatedArgIndices.removeAll(goldArgsIndices); //contains argument indices only identified by AI module
        commonGoldPredictedArgIndices.retainAll(goldArgsIndices);
        exclusiveGoldArgIndices.removeAll(highestScorePrediction.keySet());

        aiConfusionMatrix[1][1] += commonGoldPredictedArgIndices.size();
        aiConfusionMatrix[1][0] += exclusivePredicatedArgIndices.size();
        aiConfusionMatrix[0][1] += exclusiveGoldArgIndices.size();

        /*
        for (Argument goldArg : goldArgs) {
            int index = goldArg.getIndex();
            String goldLabel_str = goldArg.getType();
            int goldLabel = reverseLabelMap.get(goldLabel_str);

            if (highestScorePrediction.containsKey(index)) {
                //AI module identified this word as an argument --> increase ai_tp
                aiConfusionMatrix[1][1]++;

                if (highestScorePrediction.get(index) == goldLabel) {
                    //AC module labeled this argument correctly --> increase ac_tp
                    acConfusionMatrix.get(goldLabel)[goldLabel]++;
                } else {
                    //AC module mis-labeled this argument --> increase ac_fp
                    int prediction = highestScorePrediction.get(index);
                    acConfusionMatrix.get(prediction)[goldLabel]++;
                }
            } else {
                //AI module did not identified this word as an argument --> increase ai_fn
                aiConfusionMatrix[0][1]++;
                if (aiConfusionMatrix[0][1] < 0)
                    System.out.println(aiConfusionMatrix[0][1]);
                acConfusionMatrix.get(acConfusionMatrix.keySet().size() - 1)[goldLabel]++;
            }
        }

        //assert sentenceLength - exclusivePredicatedArgIndices.size() - aiConfusionMatrix[0][1] >= 0;
        //aiConfusionMatrix[0][0] += sentenceLength - exclusivePredicatedArgIndices.size() - aiConfusionMatrix[0][1];


        //update acConfusionMatrix for false positives
        for (int predicatedArg : highestScorePrediction.keySet()) {
            if (!goldArgsIndices.contains(predicatedArg)) {

                int predictedLabel = highestScorePrediction.get(predicatedArg);
                acConfusionMatrix.get(predictedLabel)
                        [acConfusionMatrix.get(predictedLabel).length - 1]++;
            }
        }
        */
    }


    private HashSet<Integer> getArgIndices(ArrayList<Argument> args) {
        HashSet<Integer> indices = new HashSet<Integer>();
        for (Argument arg : args)
            indices.add(arg.getIndex());
        return indices;
    }

    public void computePrecisionRecall() {
        //binary classification
        int aiTP = aiConfusionMatrix[1][1];
        int aiFP = aiConfusionMatrix[1][0];
        int aiFN = aiConfusionMatrix[0][1];

        System.out.println("AI Precision: " + (double) aiTP / (aiTP + aiFP));
        System.out.println("AI Recall: " + (double) aiTP / (aiTP + aiFN));
        System.out.println("*********************************************");

        //multi-class classification
        for (int predicatedLabel : acConfusionMatrix.keySet()) {
            int tp = acConfusionMatrix.get(predicatedLabel)[predicatedLabel]; //element on the diagonal
            int total_prediction = IntStream.of(acConfusionMatrix.get(predicatedLabel)).sum();
            int total_gold = 0;

            for (int predictedLabel : acConfusionMatrix.keySet())
                total_gold += acConfusionMatrix.get(predicatedLabel)[predicatedLabel];

            double precision = 100. * (double) tp / total_prediction;
            double recall = 100. * (double) tp / total_gold;
            System.out.println("Precision of label " + predicatedLabel + ": " + precision);
            System.out.println("Recall of label " + predicatedLabel + ": " + recall);
        }
    }


    public void predict(Sentence sentence, int aiMaxBeamSize, int acMaxBeamSize) throws Exception {
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa : pas) {

            // get best k argument assignment candidates
          //  ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = getBestAICandidates(sentence, pa, aiMaxBeamSize);

            // get best <=l argument label for each of these k assignments
          //  ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = getBestACCandidates(sentence, pa, aiCandidates, acMaxBeamSize);
          //  HashMap<Integer, Integer> highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates);

            HashMap<Integer, Integer> highestScorePrediction = getHighestScoreAISeq(sentence, pa);
            //comparing with gold argument (assuming the predicate is given) --> updates two confusion matrices
            compareWithGold(pa, highestScorePrediction);
        }

    }
}
