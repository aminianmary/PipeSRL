package SupervisedSRL;

import SentenceStruct.Argument;
import SentenceStruct.PA;
import SentenceStruct.Sentence;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.IndexMap;
import SentenceStruct.simplePA;
import util.IO;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by monadiab on 7/13/16.
 */
public class Evaluation {
    public static double evaluate(String systemOutput, ArrayList<String> goldOutputInCONLLFormat, IndexMap indexMap,
                                  HashMap<String, Integer> reverseLabelMap) throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        ArrayList<String> systemOutputInCONLLFormat = IO.readCoNLLFile(systemOutput);
        Set<String> argLabels = reverseLabelMap.keySet();

        int correctPLabel = 0;
        int wrongPLabel = 0;

        int[][] aiConfusionMatrix = new int[2][2];
        aiConfusionMatrix[0][0] = 0;
        aiConfusionMatrix[0][1] = 0;
        aiConfusionMatrix[1][0] = 0;
        aiConfusionMatrix[1][1] = 0;

        HashMap<Integer, int[]> acConfusionMatrix = new HashMap<>();
        for (int k = 0; k < argLabels.size(); k++) {
            int[] acGoldLabels = new int[argLabels.size()];
            acConfusionMatrix.put(k, acGoldLabels);
        }

        if (systemOutputInCONLLFormat.size() != goldOutputInCONLLFormat.size()) {
            System.out.print("WARNING --> Number of sentences in System output does not match with number of sentences in the Gold data");
            return -1;
        }

        for (int senIdx = 0; senIdx < systemOutputInCONLLFormat.size(); senIdx++) {
            //System.out.println("sen: "+senIdx);
            Sentence sysOutSen = new Sentence(systemOutputInCONLLFormat.get(senIdx), indexMap);
            Sentence goldSen = new Sentence(goldOutputInCONLLFormat.get(senIdx), indexMap);

            ArrayList<PA> sysOutPAs = sysOutSen.getPredicateArguments().getPredicateArgumentsAsArray();
            ArrayList<PA> goldPAs = goldSen.getPredicateArguments().getPredicateArgumentsAsArray();

            assert sysOutPAs.size()== goldPAs.size();

            for (PA goldPA : goldPAs) {
                int goldPIdx = goldPA.getPredicate().getIndex();
                String goldPLabel = goldPA.getPredicate().getPredicateGoldLabel();

                for (PA sysOutPA : sysOutPAs) {
                    int sysOutPIdx = sysOutPA.getPredicate().getIndex();
                    if (goldPIdx == sysOutPIdx) {
                        //same predicate index (predicate indices are supposed to be given)
                        String sysOutPLabel = sysOutPA.getPredicate().getPredicateGoldLabel();

                        if (goldPLabel.equals(sysOutPLabel)) {
                            //same predicate labels
                            correctPLabel++;
                            //discover argument precision/recall
                            HashMap<Integer, String> sysOutPrediction = convertPredictionToMap(sysOutPA);
                            Object[] confusionMatrices = compareWithGold(goldPA, sysOutPrediction,
                                    aiConfusionMatrix, acConfusionMatrix, reverseLabelMap);
                            aiConfusionMatrix = (int[][]) confusionMatrices[0];
                            acConfusionMatrix = (HashMap<Integer, int[]>) confusionMatrices[1];
                        } else {
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

            assert correctPLabel+ wrongPLabel == goldPAs.size();
        }
        double pdAcc= (double) correctPLabel / (correctPLabel + wrongPLabel) *100;
        System.out.println("*********************************************");
        System.out.println("Total Predicate Disambiguation Accuracy " + format.format(pdAcc));
        System.out.println("Total Number of Predicate Tokens in dev data: " + PD.totalPreds);
        System.out.println("Total Number of Unseen Predicate Tokens in dev data: " + PD.unseenPreds);
        System.out.println("*********************************************");
        return computePrecisionRecall(aiConfusionMatrix, acConfusionMatrix, reverseLabelMap);
    }


    private static Object[] compareWithGold(PA pa, HashMap<Integer, String> highestScorePrediction,
                                            int[][] aiConfusionMatrix, HashMap<Integer, int[]> acConfusionMatrix,
                                            HashMap<String, Integer> reverseLabelMap) {

        ArrayList<Argument> goldArgs = pa.getArguments();
        HashMap<Integer, String> goldArgMap = getGoldArgMap(goldArgs);
        Set<Integer> goldArgsIndices = goldArgMap.keySet();
        Set<Integer> sysOutArgIndices = getNonZeroArgs(highestScorePrediction, reverseLabelMap);


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
                //System.out.print("predictedArgIdx: "+predictedArgIdx + "\tGoldLabel: " + goldArgMap.get(predictedArgIdx) +"\n\n");
                String goldLabel = goldArgMap.get(predictedArgIdx);
                int goldLabelIdx = -1;
                if (reverseLabelMap.containsKey(goldLabel)) {
                    //seen gold label in train data
                    goldLabelIdx = reverseLabelMap.get(goldLabel);
                } else {
                    reverseLabelMap.put(goldLabel, reverseLabelMap.size());
                    goldLabelIdx = reverseLabelMap.get(goldLabel);
                    acConfusionMatrix = updateConfusionMatrix(acConfusionMatrix);
                }
                acConfusionMatrix.get(predictedLabel)[goldLabelIdx]++;

            } else {
                acConfusionMatrix.get(predictedLabel)[reverseLabelMap.get("0")]++;
            }
        }

        //update acConfusionMatrix for false negatives
        for (int goldArgIdx : goldArgMap.keySet()) {
            if (!sysOutArgIndices.contains(goldArgIdx)) {
                //ai_fn --> ac_fn
                //System.out.println(goldArgMap.get(goldArgIdx));
                String goldLabel = goldArgMap.get(goldArgIdx);
                int goldLabelIdx = -1;
                //we might see an unseen gold label at this step
                if (reverseLabelMap.containsKey(goldLabel))
                    goldLabelIdx = reverseLabelMap.get(goldArgMap.get(goldArgIdx));
                else {
                    reverseLabelMap.put(goldLabel, reverseLabelMap.size());
                    goldLabelIdx = reverseLabelMap.get(goldLabel);
                    acConfusionMatrix = updateConfusionMatrix(acConfusionMatrix);
                }
                acConfusionMatrix.get(reverseLabelMap.get("0"))
                        [goldLabelIdx]++;
            }
        }
        return new Object[]{aiConfusionMatrix, acConfusionMatrix};
    }


    public static double computePrecisionRecall(int[][] aiConfusionMatrix,
                                                HashMap<Integer, int[]> acConfusionMatrix,
                                                HashMap<String, Integer> reverseLabelMap) {
        DecimalFormat format = new DecimalFormat("##.00");
        //binary classification
        int aiTP = aiConfusionMatrix[1][1];
        int aiFP = aiConfusionMatrix[1][0];
        int aiFN = aiConfusionMatrix[0][1];
        int total_ai_predictions = aiTP + aiFP;

        System.out.println("Total AI prediction " + total_ai_predictions);
        System.out.println("AI Precision: " + format.format((double) aiTP / (aiTP + aiFP)* 100));
        System.out.println("AI Recall: " + format.format((double) aiTP / (aiTP + aiFN) * 100));
        System.out.println("*********************************************");

        String[] labelMap = new String[reverseLabelMap.size()];
        for (String label : reverseLabelMap.keySet())
            labelMap[reverseLabelMap.get(label)] = label;

        int total_ac_predictions = 0;
        int total_tp = 0;
        int total_gold = 0;

        //multi-class classification
        for (int predicatedLabel : acConfusionMatrix.keySet()) {
            if (predicatedLabel != reverseLabelMap.get("0")) {
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

                //double precision = 100. * (double) tp / total_prediction_4_this_label;
                //double recall = 100. * (double) tp / total_gold_4_this_label;
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
        return FScore;
    }


    public static double computePrecisionRecall(int[][] aiConfusionMatrix) {
        DecimalFormat format = new DecimalFormat("##.00");
        //binary classification
        int aiTP = aiConfusionMatrix[1][1];
        int aiFP = aiConfusionMatrix[1][0];
        int aiFN = aiConfusionMatrix[0][1];
        int total_ai_predictions = aiTP + aiFP;

        double precision = ((double) aiTP / (aiTP + aiFP) ) * 100;
        double recall = ((double) aiTP / (aiTP + aiFN)) *100;
        System.out.println("Total AI prediction " + total_ai_predictions);
        System.out.println("AI Precision: " + format.format(precision));
        System.out.println("AI Recall: " + format.format(recall));
        double fscore = (2 * precision * recall) / (precision + recall);
        System.out.println("AI F1-score: " + format.format(fscore));
        System.out.println("*********************************************");
        return fscore;
    }


    public static int[][] evaluateAI4ThisSentence(Sentence goldSentence, HashMap<Integer, simplePA> prediction,
                                                  int[][] aiConfusionMatrix) {
        ArrayList<PA> goldPAs = goldSentence.getPredicateArguments().getPredicateArgumentsAsArray();

        for (PA goldPA : goldPAs) {
            int goldPIdx = goldPA.getPredicate().getIndex();
            for (int sysOutPIdx : prediction.keySet()) {
                if (goldPIdx == sysOutPIdx) {
                    HashMap<Integer, String> prediction4ThisPredicate = prediction.get(sysOutPIdx).getArgumentLabels();

                    ArrayList<Argument> goldArgs = goldPA.getArguments();
                    HashMap<Integer, String> goldArgMap = getGoldArgMap(goldArgs);
                    Set<Integer> goldArgsIndices = goldArgMap.keySet();
                    Set<Integer> sysOutArgIndices = prediction4ThisPredicate.keySet();

                    HashSet<Integer> exclusiveGoldArgIndices = new HashSet(goldArgsIndices);
                    HashSet<Integer> commonGoldPredictedArgIndices = new HashSet(sysOutArgIndices);
                    HashSet<Integer> exclusivePredicatedArgIndices = new HashSet(sysOutArgIndices);

                    exclusivePredicatedArgIndices.removeAll(goldArgsIndices); //contains argument indices only identified by AI module
                    commonGoldPredictedArgIndices.retainAll(goldArgsIndices);
                    exclusiveGoldArgIndices.removeAll(sysOutArgIndices);

                    aiConfusionMatrix[1][1] += commonGoldPredictedArgIndices.size();
                    aiConfusionMatrix[1][0] += exclusivePredicatedArgIndices.size();
                    aiConfusionMatrix[0][1] += exclusiveGoldArgIndices.size();
                }
                break;
            }
        }
        return aiConfusionMatrix;
    }

    public static void evaluatePD (ArrayList<String> trainSentences,ArrayList<String> devSentences, ArrayList<String> goldEvalSentences,
                                   HashMap<Integer, String>[] pdPredictionsOnEvalData, IndexMap indexMap, int numOfPDFeatures)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");
        int seenLemm_seenSense_correct =0;
        int seenLemm_unseenSense_correct =0;
        int unseenLemma_correct =0;

        int seenLemm_seenSense_total =0;
        int seenLemm_unseenSense_total =0;
        int unseenLemma_total =0;

        int correct =0;
        int total =0;

        HashMap<Integer, HashMap<String, HashSet<Object[]>>> trainPredicateLexicon =
                PD.buildPredicateLexicon(trainSentences, indexMap, numOfPDFeatures);

        HashMap<Integer, HashMap<String, HashSet<Object[]>>> devPredicateLexicon =
                PD.buildPredicateLexicon(devSentences, indexMap, numOfPDFeatures);

        for (int senID =0 ; senID < goldEvalSentences.size() ; senID++){
            Sentence sentence = new Sentence(goldEvalSentences.get(senID), indexMap );
            int[] sentenceLemmas = sentence.getLemmas();
            String[] sentenceLemmaStr = sentence.getLemmas_str();
            HashMap<Integer, String> goldPredicateLabelMap = sentence.getPredicatesGoldLabelMap();
            HashMap<Integer, String> predicatedPredicateLabelMap = pdPredictionsOnEvalData[senID];
            assert goldPredicateLabelMap.size() == predicatedPredicateLabelMap.size();
            total += goldPredicateLabelMap.size();

            for (int pIdx: goldPredicateLabelMap.keySet()) {
                int predicateLemma = sentenceLemmas[pIdx];
                String goldPredicateLabel = goldPredicateLabelMap.get(pIdx);
                String predictedPredicateLabel =predicatedPredicateLabelMap.get(pIdx);

                if (goldPredicateLabel.equals(predictedPredicateLabel))
                    correct++;

                if (trainPredicateLexicon.containsKey(predicateLemma)){
                    //seen lemma
                    if (trainPredicateLexicon.get(predicateLemma).containsKey(goldPredicateLabel))
                    {
                        //seen sense
                        seenLemm_seenSense_total++;
                        if (goldPredicateLabel.equals(predictedPredicateLabel))
                            seenLemm_seenSense_correct++;
                        System.out.print("SEEN LEMMA-SEEN SENSE\tLemma index:\t"+predicateLemma+"\tLemma:\t"+
                                indexMap.int2str(predicateLemma)+"\tGold label:\t"+goldPredicateLabel+
                                "\tAutomatic label:\t"+predictedPredicateLabel+"\tSeen labels in train:\t"+trainPredicateLexicon.get(predicateLemma).keySet()+"\t");
                        if (devPredicateLexicon.containsKey(predicateLemma)) {
                            System.out.print("Seen in dev with labels:\t" + devPredicateLexicon.get(predicateLemma).keySet() +"\n");
                        }
                        else
                            System.out.print("Not seen in dev data");
                    }else{
                        //unseen sense
                        seenLemm_unseenSense_total++;
                        if (goldPredicateLabel.equals(predictedPredicateLabel)) {
                            seenLemm_unseenSense_correct++;
                        }
                    }
                }else{
                    //unseen lemma
                    unseenLemma_total++;
                    if (goldPredicateLabel.equals(predictedPredicateLabel))
                        unseenLemma_correct++;
                }
            }
        }

        double seenLemma_seenSense_acc = ((double) seenLemm_seenSense_correct/seenLemm_seenSense_total) * 100;
        double seenLemma_unseenSense_acc = ((double) seenLemm_unseenSense_correct/seenLemm_unseenSense_total) * 100;
        double unseenLemma_acc = ((double) unseenLemma_correct/unseenLemma_total) * 100;
        double total_acc = ((double) correct/total) * 100;

        System.out.print("\nPD Accuracy on seenLemma_seenSense: " + seenLemm_seenSense_correct + "/" + seenLemm_seenSense_total + "= " + format.format(seenLemma_seenSense_acc)+"\n");
        System.out.print("\nPD Accuracy on seenLemma_unseenSense: " + seenLemm_unseenSense_correct + "/" + seenLemm_unseenSense_total +"= " + format.format(seenLemma_unseenSense_acc)+"\n");
        System.out.print("\nPD Accuracy on unseenLemma: " + unseenLemma_correct +"/" + unseenLemma_total +"= " + format.format(unseenLemma_acc)+"\n");
        System.out.print("\nPD Accuracy: " + correct +"/"+total +"= "+format.format(total_acc)+"\n");

    }

    //////////// SUPPORTING FUNCTIONS /////////////////////////////////////////////////

    private static HashMap<Integer, String> getGoldArgMap(ArrayList<Argument> args) {
        HashMap<Integer, String> goldArgMap = new HashMap<Integer, String>();
        for (Argument arg : args)
            goldArgMap.put(arg.getIndex(), arg.getType());
        return goldArgMap;
    }

    private static HashSet<Integer> getNonZeroArgs(HashMap<Integer, String> prediction, HashMap<String, Integer> reverseLabelMap) {
        HashSet<Integer> nonZeroArgs = new HashSet();
        for (int key : prediction.keySet())
            if (!prediction.get(key).equals(reverseLabelMap.get("0")))
                nonZeroArgs.add(key);

        return nonZeroArgs;
    }

    private static HashMap<Integer, String> convertPredictionToMap(PA pa) {
        HashMap<Integer, String> highestScorePrediction = new HashMap<>();
        ArrayList<Argument> args = pa.getArguments();
        for (Argument arg : args)
            highestScorePrediction.put(arg.getIndex(), arg.getType());
        return highestScorePrediction;
    }

    private static HashMap<Integer, int[]> updateConfusionMatrix(HashMap<Integer, int[]> currentConfusionMatrix) {
        HashMap<Integer, int[]> newConfusionMatrix = new HashMap<Integer, int[]>();
        for (int predictedLabel : currentConfusionMatrix.keySet()) {
            int[] currentGoldLabels = currentConfusionMatrix.get(predictedLabel);
            int[] newGoldLabels = new int[currentGoldLabels.length + 1];
            for (int i = 0; i < currentGoldLabels.length; i++)
                newGoldLabels[i] = currentGoldLabels[i];
            newGoldLabels[currentGoldLabels.length] = 0;
            newConfusionMatrix.put(predictedLabel, newGoldLabels);
        }
        int[] temp = new int[currentConfusionMatrix.size() + 1];
        newConfusionMatrix.put(currentConfusionMatrix.size(), temp);

        return newConfusionMatrix;
    }

}
