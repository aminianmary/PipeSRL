package ml;

import SupervisedSRL.Strcutures.CompactArray;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 3/18/15
 * Time: 11:15 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */
public class AveragedPerceptron implements Serializable {
    public int correct = 0;
    public int[][] confusionMatrix = new int[2][2];
    private HashMap<Object, CompactArray>[] weights;
    private HashMap<Object, CompactArray>[] avgWeights;
    private String[] labelMap;
    private HashMap<String, Integer> reverseLabelMap;
    private int iteration;

    public AveragedPerceptron(HashSet<String> possibleLabels, int featureTemplateSize) {
        this.iteration = 1;
        weights = new HashMap[featureTemplateSize];
        avgWeights = new HashMap[featureTemplateSize];

        for (int i = 0; i < featureTemplateSize; i++) {
            weights[i] = new HashMap<>();
            avgWeights[i] = new HashMap<>();
        }
        labelMap = new String[possibleLabels.size()];
        reverseLabelMap = new HashMap<>();
        int i = 0;
        for (String label : possibleLabels) {
            labelMap[i] = label;
            reverseLabelMap.put(label, i);
            i++;
        }
        confusionMatrix = new int[2][2];
    }


    public AveragedPerceptron(HashMap<Object, CompactArray>[] avgWeights, String[] labelMap,
                              HashMap<String, Integer> reverseLabelMap) {
        this.avgWeights = avgWeights;
        this.labelMap = labelMap;
        this.reverseLabelMap = reverseLabelMap;
    }


    public static AveragedPerceptron loadModel(String filePath) throws Exception {
        FileInputStream fis = new FileInputStream(filePath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);
        HashMap<Object, CompactArray>[] newAvgWeight =
                (HashMap<Object, CompactArray>[]) reader.readObject();
        String[] labelMap = (String[]) reader.readObject();
        HashMap<String, Integer> reverseLabelMap = (HashMap<String, Integer>) reader.readObject();
        fis.close();
        gz.close();
        reader.close();
        return new AveragedPerceptron(newAvgWeight, labelMap, reverseLabelMap);

    }

    public HashMap<Object, CompactArray>[] getWeights() {
        return weights;
    }

    public HashMap<Object, CompactArray>[] getAvgWeights() {
        return avgWeights;
    }

    public int getIteration() {
        return iteration;
    }

    public String[] getLabelMap() {
        return labelMap;
    }

    public HashMap<String, Integer> getReverseLabelMap() {
        return reverseLabelMap;
    }

    public void learnInstance(Object[] features, String label, double completeness) {
        int argmax = argmax(features, false);
        int gold = reverseLabelMap.get(label);
        if (argmax != gold) {
            updateWeight(argmax, gold, features, completeness);
            if (reverseLabelMap.size() == 2)
                confusionMatrix[argmax][gold]++;
        } else {
            correct++;
            if (reverseLabelMap.size() == 2)
                confusionMatrix[gold][gold]++;
        }
        iteration++;
    }

    private void updateWeight(int argmax, int gold, Object[] features, double completeness) {
        for (int i = 0; i < features.length; i++) {
            updateWeight(argmax, i, features[i], -1 * completeness);
            updateWeight(gold, i, features[i], 1 * completeness);
        }
    }

    private void updateWeight(int label, int featIndex, Object feature, double change) {
        if (!weights[featIndex].containsKey(feature)) {
            double[] tempArray1 = new double[1];
            tempArray1[0] = change;
            CompactArray subWeights = new CompactArray(label, tempArray1);
            weights[featIndex].put(feature, subWeights);

            double[] tempArray2 = new double[1];
            tempArray2[0] = iteration * change;
            CompactArray avgSubWeights = new CompactArray(label, tempArray2);
            avgWeights[featIndex].put(feature, avgSubWeights);
        } else {
            CompactArray subWeights = weights[featIndex].get(feature);
            subWeights.expandArray(label, change);

            CompactArray avgSubWeights = avgWeights[featIndex].get(feature);
            avgSubWeights.expandArray(label, iteration * change);
        }
    }

    public String predict(Object[] features) {
        return labelMap[argmax(features, true)];
    }

    private int argmax(Object[] features, boolean decode) {
        HashMap<Object, CompactArray>[] map = decode ? avgWeights : weights;
        double max = Double.NEGATIVE_INFINITY;
        int argmax = 0;

        double[] score = new double[labelMap.length];

        for (int f = 0; f < features.length; f++) {
            if (map[f].containsKey(features[f])) {
                CompactArray w = map[f].get(features[f]);
                int offset = w.getOffset();
                for (int i = 0; i < w.length(); i++)
                    score[i + offset] += w.getArray()[i];
            }
        }

        for (int i = 0; i < score.length; i++) {
            if (score[i] > max) {
                argmax = i;
                max = score[i];
            }
        }

        return argmax;
    }

    public double[] score(Object[] features) throws Exception {
        double[] score = new double[labelMap.length];

        for (int f = 0; f < features.length; f++) {
            if (avgWeights[f].containsKey(features[f])) {
                CompactArray w = avgWeights[f].get(features[f]);
                int offset = w.getOffset();
                for (int i = 0; i < w.length(); i++)
                    score[i + offset] += w.getArray()[i];
            }
        }

        double[] logProbs = new double[labelMap.length];
        if (labelMap.length > 2) {
            //use softmax
            double sumOfScores = 0;
            double maxScore = Double.NEGATIVE_INFINITY;
            for (double s : score) {
                if (s > maxScore)
                    maxScore = s;
            }

            for (double s : score)
                sumOfScores += Math.exp(s - maxScore);

            for (int i = 0; i < score.length; i++)
                logProbs[i] = score[i] - maxScore - Math.log(sumOfScores);

        } else if (labelMap.length == 2) {
            //use logit function
            double score1 = (labelMap[0].equals("1")) ? score[0] : score[1];
            double prob1 = 1.0 / (1 + Math.exp(-score1));
            double prob0 = 1 - prob1;
            if (labelMap[0].equals("1")) {
                logProbs[0] = Math.log(prob1);
                logProbs[1] = Math.log(prob0);
            } else {
                logProbs[0] = Math.log(prob0);
                logProbs[1] = Math.log(prob1);
            }
        } else
            throw new Exception("Less than 2 labels!");

        return logProbs;
    }

    public void saveModel(String filePath) throws Exception {

        HashMap<Object, CompactArray>[] newAvgMap = new HashMap[weights.length];

        for (int f = 0; f < weights.length; f++) {
            newAvgMap[f] = new HashMap<>();
            for (Object feat : weights[f].keySet()) {
                double[] w = weights[f].get(feat).getArray();
                double[] aw = avgWeights[f].get(feat).getArray();
                double[] naw = new double[w.length];
                for (int i = 0; i < w.length; i++) {
                    naw[i] = w[i] - (aw[i] / iteration);
                }
                CompactArray nawCompact = new CompactArray(weights[f].get(feat).getOffset(), naw);
                newAvgMap[f].put(feat, nawCompact);
            }
        }

        FileOutputStream fos = new FileOutputStream(filePath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);
        writer.writeObject(newAvgMap);
        writer.writeObject(labelMap);
        writer.writeObject(reverseLabelMap);
        writer.close();
    }

    public AveragedPerceptron calculateAvgWeights() {
        HashMap<Object, CompactArray>[] newAvgMap = new HashMap[weights.length];

        for (int f = 0; f < weights.length; f++) {
            newAvgMap[f] = new HashMap<>();
            for (Object feat : weights[f].keySet()) {
                double[] w = weights[f].get(feat).getArray();
                double[] aw = avgWeights[f].get(feat).getArray();
                double[] naw = new double[w.length];
                for (int i = 0; i < w.length; i++) {
                    naw[i] = w[i] - (aw[i] / iteration);
                }
                CompactArray nawCompact = new CompactArray(weights[f].get(feat).getOffset(), naw);
                newAvgMap[f].put(feat, nawCompact);
            }
        }
        return new AveragedPerceptron(newAvgMap, labelMap, reverseLabelMap);
    }
}