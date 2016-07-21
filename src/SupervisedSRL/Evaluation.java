package SupervisedSRL;

import Sentence.Argument;
import Sentence.PA;
import Sentence.Sentence;
import SupervisedSRL.Strcutures.IndexMap;
import util.IO;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by monadiab on 7/13/16.
 */
public class Evaluation {


    public static void evaluate (String systemOutput, String goldOutput, IndexMap indexMap, Set<String> argLabels) throws IOException
    {
        DecimalFormat format = new DecimalFormat("##.00");

        List<String> systemOutputInCONLLFormat = IO.readCoNLLFile(systemOutput);
        List<String> goldOutputInCONLLFormat = IO.readCoNLLFile(goldOutput);

        int correctPLabel =0;
        int wrongPLabel =0;

        int[][] aiConfusionMatrix = new int[2][2];
        aiConfusionMatrix[0][0] = 0;
        aiConfusionMatrix[0][1] = 0;
        aiConfusionMatrix[1][0] = 0;
        aiConfusionMatrix[1][1] = 0;

        HashMap<Integer, int[]> acConfusionMatrix = new HashMap<Integer, int[]>();
        for (int k = 0; k < argLabels.size(); k++) {
            int[] acGoldLabels = new int[argLabels.size()];
            acConfusionMatrix.put(k, acGoldLabels);
        }

        HashMap<String, Integer> reverseLabelMap = createReverseLabelMap(argLabels);

        if (systemOutputInCONLLFormat.size() != goldOutputInCONLLFormat.size()) {
            System.out.print("WARNING --> Number of sentences in System output does not match with number of sentences in the Gold data");
            return;
        }

        boolean decode = true;
        for (int senIdx =0; senIdx< systemOutputInCONLLFormat.size(); senIdx++)
        {
            Sentence sysOutSen = new Sentence(systemOutputInCONLLFormat.get(senIdx), indexMap, decode);
            Sentence goldSen = new Sentence(goldOutputInCONLLFormat.get(senIdx), indexMap, decode);

            ArrayList<PA> sysOutPAs = sysOutSen.getPredicateArguments().getPredicateArgumentsAsArray();
            ArrayList<PA> goldPAs = goldSen.getPredicateArguments().getPredicateArgumentsAsArray();

            for (PA goldPA : goldPAs)
            {
                int goldPIdx = goldPA.getPredicateIndex();
                String goldPLabel = goldPA.getPredicateLabel();
                for (PA sysOutPA : sysOutPAs)
                {
                    int sysOutPIdx=  sysOutPA.getPredicateIndex();
                    if (goldPIdx == sysOutPIdx)
                    {
                        //same predicate index (predicate indices are supposed to be given)
                        String sysOutPLabel = sysOutPA.getPredicateLabel();
                        if (goldPLabel.equals(sysOutPLabel)) {
                            //same predicate labels
                            correctPLabel++;
                            //discover argument precision/recall
                            HashMap<Integer, String> sysOutPrediction = convertPredictionToMap(sysOutPA);
                            Object[] confusionMatrices = compareWithGold(goldPA, sysOutPrediction,
                                    aiConfusionMatrix, acConfusionMatrix, reverseLabelMap);
                            aiConfusionMatrix = (int[][]) confusionMatrices[0];
                            acConfusionMatrix = (HashMap<Integer, int[]>) confusionMatrices[1];
                        }else
                        {
                            //different predicate labels
                            wrongPLabel++;
                            //discover argument precision/recall
                            HashMap<Integer, String> sysOutPrediction = convertPredictionToMap(sysOutPA);
                            Object[] confusionMatrices = compareWithGold(goldPA, sysOutPrediction,
                                    aiConfusionMatrix, acConfusionMatrix, reverseLabelMap);
                            aiConfusionMatrix = (int[][]) confusionMatrices[0];
                            acConfusionMatrix = (HashMap<Integer, int[]>) confusionMatrices[1];
                        }
                        break;
                    }
                }
            }
        }
        System.out.println("*********************************************");
        System.out.println("Total Predicate Disambiguation Accuracy " + format.format((double) correctPLabel / (correctPLabel+wrongPLabel)));
        System.out.println("*********************************************");
        computePrecisionRecall(aiConfusionMatrix, acConfusionMatrix, reverseLabelMap);
    }


    private static Object[] compareWithGold(PA pa, HashMap<Integer, String> highestScorePrediction,
                                            int[][] aiConfusionMatrix, HashMap<Integer, int[]> acConfusionMatrix,
                                            HashMap<String, Integer> reverseLabelMap) {

        ArrayList<Argument> goldArgs = pa.getArguments();
        HashMap<Integer, String> goldArgMap = getGoldArgMap(goldArgs);
        Set<Integer> goldArgsIndices = goldArgMap.keySet();
        Set<Integer> sysOutArgIndices = getNonZeroArgs(highestScorePrediction);


        HashSet<Integer> exclusiveGoldArgIndices = new HashSet(goldArgsIndices);
        HashSet<Integer> commonGoldPredictedArgIndices = new HashSet(sysOutArgIndices);
        HashSet<Integer> exclusivePredicatedArgIndices = new HashSet(sysOutArgIndices);

        exclusivePredicatedArgIndices.removeAll(goldArgsIndices); //contains argument indices only identified by AI module
        commonGoldPredictedArgIndices.retainAll(goldArgsIndices);
        exclusiveGoldArgIndices.removeAll(sysOutArgIndices);

        aiConfusionMatrix[1][1] += commonGoldPredictedArgIndices.size();
        aiConfusionMatrix[1][0] += exclusivePredicatedArgIndices.size();
        aiConfusionMatrix[0][1] += exclusiveGoldArgIndices.size();

        for (int predictedArgIdx : sysOutArgIndices) {
            int predictedLabel = reverseLabelMap.get(highestScorePrediction.get(predictedArgIdx));
            if (goldArgMap.containsKey(predictedArgIdx)) {
                int goldLabel = reverseLabelMap.get(goldArgMap.get(predictedArgIdx));
                acConfusionMatrix.get(predictedLabel)[goldLabel]++;
            } else {
                acConfusionMatrix.get(predictedLabel)[reverseLabelMap.get("0")]++;
            }
        }

        //update acConfusionMatrix for false negatives
        for (int goldArgIdx : goldArgMap.keySet()) {
            if (!sysOutArgIndices.contains(goldArgIdx)) {
                //ai_fn --> ac_fn
                int goldLabel = reverseLabelMap.get(goldArgMap.get(goldArgIdx));
                acConfusionMatrix.get(reverseLabelMap.get("0"))
                        [goldLabel]++;
            }
        }
        return new Object[]{aiConfusionMatrix, acConfusionMatrix};
    }

    private static void computePrecisionRecall(int[][] aiConfusionMatrix,
                                               HashMap<Integer, int[]> acConfusionMatrix,
                                               HashMap<String, Integer> reverseLabelMap) {
        DecimalFormat format = new DecimalFormat("##.00");
        //binary classification
        int aiTP = aiConfusionMatrix[1][1];
        int aiFP = aiConfusionMatrix[1][0];
        int aiFN = aiConfusionMatrix[0][1];
        int total_ai_predictions = aiTP + aiFP;

        System.out.println("Total AI prediction " + total_ai_predictions);
        System.out.println("AI Precision: " + format.format((double) aiTP / (aiTP + aiFP)));
        System.out.println("AI Recall: " + format.format((double) aiTP / (aiTP + aiFN)));
        System.out.println("*********************************************");

        String[] labelMap = new String[reverseLabelMap.size()];
        for (String label : reverseLabelMap.keySet())
            labelMap[reverseLabelMap.get(label)] = label;

        int total_ac_predictions = 0;
        int total_tp = 0;
        int total_gold = 0;

        //multi-class classification
        for (int predicatedLabel : acConfusionMatrix.keySet())
        {
            if (predicatedLabel!=reverseLabelMap.get("0"))
            {
                //for real arguments
                int tp = acConfusionMatrix.get(predicatedLabel)[predicatedLabel]; //element on the diagonal
                total_tp += tp;

                int total_prediction_4_this_label = 0;
                for (int element : acConfusionMatrix.get(predicatedLabel))
                    total_prediction_4_this_label += element;

                total_ac_predictions += total_prediction_4_this_label;

                int total_gold_4_this_label = 0;

                for (int pLabel : acConfusionMatrix.keySet())
                    total_gold_4_this_label += acConfusionMatrix.get(pLabel)[predicatedLabel];

                total_gold += total_gold_4_this_label;

                double precision = 100. * (double) tp / total_prediction_4_this_label;
                double recall = 100. * (double) tp / total_gold_4_this_label;
                System.out.println("Precision of label " + labelMap[predicatedLabel] + ": " + format.format(precision));
                System.out.println("Recall of label " + labelMap[predicatedLabel] + ": " + format.format(recall));
            }
        }

        System.out.println("*********************************************");
        System.out.println("Total AC prediction " + format.format(total_ac_predictions));
        System.out.println("Total number of tp: " + format.format(total_tp));

        double micro_precision = 100. * (double) total_tp / total_ac_predictions;
        double micro_recall = 100. * (double) total_tp / total_gold;
        double FScore = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall);

        System.out.println("Micro Precision: " + format.format(micro_precision));
        System.out.println("Micro Recall: " + format.format(micro_recall));
        System.out.println("Averaged F1-score: " + format.format(FScore));
    }

    /*
    private static void compareWithGoldJoint(PA pa, HashMap<Integer, Integer> highestScorePrediction) {

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
    */

    //////////// SUPPORTING FUNCTIONS /////////////////////////////////////////////////

    private static HashMap<Integer, String> getGoldArgMap(ArrayList<Argument> args) {
        HashMap<Integer, String> goldArgMap= new HashMap<Integer, String>();
        for (Argument arg : args)
            goldArgMap.put(arg.getIndex(), arg.getType());
        return goldArgMap;
    }


    private static HashSet<Integer> getNonZeroArgs (HashMap<Integer, String> prediction)
    {
        HashSet<Integer> nonZeroArgs = new HashSet();
        for (int key: prediction.keySet())
            if (!prediction.get(key).equals("0"))
                nonZeroArgs.add(key);

        return nonZeroArgs;
    }


    private static HashSet<Integer> getPredicateIndices (ArrayList<PA> predicateArguments)
    {
        HashSet<Integer> predicateIndices =  new HashSet<Integer>(  );
        for (PA pa : predicateArguments)
            predicateIndices.add(pa.getPredicateIndex());
        return predicateIndices;
    }


    private static HashMap<Integer, String> convertPredictionToMap(PA pa)
    {
        HashMap<Integer, String> highestScorePrediction = new HashMap<Integer, String>();
        ArrayList<Argument> args = pa.getArguments();
        for (Argument arg: args)
            highestScorePrediction.put(arg.getIndex(), arg.getType());
        return highestScorePrediction;
    }

    private static HashMap<String, Integer> createReverseLabelMap (Set<String> argLabels)
    {
        HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>();
        int index = 0;
        for (String label:argLabels)
            reverseLabelMap.put(label, index++);
        return reverseLabelMap;
    }
}
