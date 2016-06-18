package ml;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 3/18/15
 * Time: 11:15 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */
public class AveragedPerceptron  implements Serializable {
    // todo change every String to Object
    private HashMap<String, double[]>[] weights;
    private HashMap<String, double[]>[] avgWeights;

    // todo Keep these as strings
    private String[] labelMap;
    private HashMap<String, Integer> reverseLabelMap;
    private int iteration;
    public int correct = 0;

    public int[][] confusionMatrix = new int[2][2];

    public AveragedPerceptron(HashSet<String> possibleLabels, int featureTemplateSize) {
        this.iteration = 1;
        weights = new HashMap[featureTemplateSize];
        avgWeights = new HashMap[featureTemplateSize];

        for(int i=0;i<featureTemplateSize;i++){
            weights[i] = new HashMap<String, double[]>();
            avgWeights[i] = new HashMap<String, double[]>();
        }
        labelMap = new String[possibleLabels.size()];
        reverseLabelMap = new HashMap<String, Integer>();
        int i=0;
        for (String label:possibleLabels) {
            labelMap[i] = label;
            reverseLabelMap.put(label, i);
            i++;
        }
        confusionMatrix = new int[2][2];
    }


    private AveragedPerceptron(HashMap<String, double[]>[] avgWeights, String[] labelMap,
                               HashMap<String, Integer> reverseLabelMap) {
        this.avgWeights = avgWeights;
        this.labelMap = labelMap;
        this.reverseLabelMap = reverseLabelMap;
    }

    public String[] getLabelMap() {return labelMap;}

    public HashMap<String, Integer> getReverseLabelMap() {return reverseLabelMap;}


    public void learnInstance(String[] features, String label) {
        int argmax = argmax(features, false);
        int gold = reverseLabelMap.get(label);
        if (argmax != gold) {
            updateWeight(argmax, gold, features);
            if (reverseLabelMap.size() == 2)
                confusionMatrix[argmax][gold]++;
        } else {
            correct++;
            if (reverseLabelMap.size() == 2)
                confusionMatrix[gold][gold]++;
        }
        iteration++;
    }

    private void updateWeight(int argmax, int gold, String[] features) {
        for (int i=0;i<features.length;i++) {
            updateWeight(argmax,i, features[i], -1);
            updateWeight(gold,i, features[i], 1);
        }
    }

    private void updateWeight(int label, int featIndex, String feature, double change) {
        if (!weights[featIndex].containsKey(feature)) {
            double[] subWeights = new double[labelMap.length];
            subWeights[label] = change;
            weights[featIndex].put(feature, subWeights);

            double[] avgSubWeights = new double[labelMap.length];
            avgSubWeights[label] = iteration * change;
            avgWeights[featIndex].put(feature, avgSubWeights);
        } else {
            double[] subWeights = weights[featIndex].get(feature);
            subWeights[label] += change;

            double[] avgSubWeights = avgWeights[featIndex].get(feature);
            avgSubWeights[label] += iteration * change;
        }
    }


    public String predict(String[] features) {
        return labelMap[argmax(features, true)];
    }

    private int argmax(String[] features, boolean decode) {
        HashMap<String, double[]>[] map = decode ? avgWeights : weights;
        double max = Double.NEGATIVE_INFINITY;
        int argmax = 0;

        double[] score = new double[labelMap.length];

        for (int f=0;f<features.length;f++) {
            if (map[f].containsKey(features[f])) {
                double[] w = map[f].get(features[f]);
                for (int i = 0; i < w.length; i++)
                    score[i] += w[i];
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

    public double[] score(String[] features) {
        double[] score = new double[labelMap.length];

        for (int f=0;f<features.length;f++) {
            if (avgWeights[f].containsKey(features[f])) {
                double[] w = avgWeights[f].get(features[f]);
                for (int i = 0; i < w.length; i++)
                    score[i] += w[i];
            }
        }
        return score;
    }

    // todo can copy this to the new class "ModelInformation" and just add writeObject(indexMaps)
    public void saveModel(String filePath) throws Exception {
        HashMap<String, double[]>[] newAvgMap = new HashMap[weights.length];

        for(int f=0;f<weights.length;f++) {
            newAvgMap[f] = new HashMap<String, double[]>();
            for (String feat : weights[f].keySet()) {
                double[] w = weights[f].get(feat);
                double[] aw = avgWeights[f].get(feat);
                double[] naw = new double[labelMap.length];
                for (int i = 0; i < labelMap.length; i++) {
                    naw[i] = w[i] - (aw[i] / iteration);
                }
                newAvgMap[f].put(feat, naw);
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

    public static AveragedPerceptron loadModel(String filePath) throws Exception {
        FileInputStream fis = new FileInputStream(filePath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);
        HashMap<String, double[]>[] newAvgWeight =
                (HashMap<String, double[]>[]) reader.readObject();
        String[] labelMap = (String[]) reader.readObject();
        HashMap<String, Integer> reverseLabelMap = (HashMap<String, Integer>) reader.readObject();

        fis.close();
        gz.close();
        reader.close();

        return new AveragedPerceptron(newAvgWeight, labelMap, reverseLabelMap);

    }

}