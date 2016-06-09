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
    private HashMap<String, Double>[] weights;
    private HashMap<String, Double>[] avgWeights;
    private String[] labelMap;
    private HashMap<String, Integer> reverseLabelMap;
    private int iteration;
    public int correct = 0;

    public int[][] confusionMatrix = new int[2][2];

    public AveragedPerceptron(HashSet<String> possibleLabels) {
        this.iteration = 1;
        weights = new HashMap[possibleLabels.size()];
        avgWeights = new HashMap[possibleLabels.size()];
        labelMap = new String[possibleLabels.size()];
        reverseLabelMap = new HashMap<String, Integer>();
        int i=0;
        for (String label:possibleLabels) {
            weights[i] = new HashMap<String, Double>();
            avgWeights[i] = new HashMap<String, Double>();
            labelMap[i] = label;
            reverseLabelMap.put(label, i);
            i++;
        }
        confusionMatrix = new int[2][2];
    }


    private AveragedPerceptron(HashMap<String, Double>[] avgWeights, String[] labelMap,
                               HashMap<String, Integer> reverseLabelMap) {
        this.avgWeights = avgWeights;
        this.labelMap = labelMap;
        this.reverseLabelMap = reverseLabelMap;
    }

    public String[] getLabelMap() {return labelMap;}

    public HashMap<String, Integer> getReverseLabelMap() {return reverseLabelMap;}


    public void learnInstance(List<String> features, String label) {
        int argmax = argmax(features, false);
        int gold = reverseLabelMap.get(label);
        if (argmax != gold) {
            updateWeight(argmax, gold, features);
            if(reverseLabelMap.size()==2)
            confusionMatrix[argmax][gold]++;
        }else {
            correct++;
            if(reverseLabelMap.size()==2)
                confusionMatrix[gold][gold]++;
        }
        iteration++;
    }

    public void learnInstance(List<String> features,HashMap<String,Double> realValuedFeatures, String label) {
        int argmax = argmax(features,realValuedFeatures, false);
        int gold = reverseLabelMap.get(label);
        if (argmax != gold)
            updateWeight(argmax, gold, features, realValuedFeatures);
        else
            correct++;
        iteration++;
    }

    private void updateWeight(int argmax, int gold, List<String> features) {
        for (String feat : features) {
            updateWeight(argmax, feat, -1);
            updateWeight(gold, feat, 1);
        }
    }

    private void updateWeight(int argmax, int gold, List<String> features, HashMap<String,Double> realValueFeatures) {
        for (String feat : features) {
            updateWeight(argmax, feat, -1);
            updateWeight(gold, feat, 1);
        }

        for (String feat : realValueFeatures.keySet()) {
            double val = realValueFeatures.get(feat);
            updateWeight(argmax, feat, -1*val);
            updateWeight(gold, feat, 1* val);
        }
    }

    private void updateWeight(int label, String feature, double change) {
        if (!weights[label].containsKey(feature)) {
            weights[label].put(feature, change);
            avgWeights[label].put(feature, iteration * change);
        } else {
            weights[label].put(feature, weights[label].get(feature) + change);
            avgWeights[label].put(feature, avgWeights[label].get(feature) + iteration * change);
        }
    }


    public String predict(List<String> features) {
        return labelMap[argmax(features, true)];
    }

    public String predict(List<String> features, HashMap<String,Double> realValuedFeatures) {
        return labelMap[argmax(features,realValuedFeatures, true)];
    }

    private int argmax(List<String> features, boolean decode) {
        HashMap<String, Double>[] map = decode ? avgWeights : weights;
        double max = Double.NEGATIVE_INFINITY;
        int argmax = 0;

        for (int i = 0; i < map.length; i++) {
            double score = 0.0;

            HashMap<String, Double> w = map[i];
            for (String feat : features) {
                if (w.containsKey(feat))
                    score += w.get(feat);
            }

            if (score > max) {
                max = score;
                argmax = i;
            }

        }

        return argmax;
    }

    public double score(List<String> features, String label) {
        double score = 0.0;

        HashMap<String, Double> w = avgWeights[reverseLabelMap.get(label)];
        for (String feat : features) {
            if (w.containsKey(feat))
                score += w.get(feat);
        }

        return score;
    }

    //TODO check if this function works properly -- also check with Sadegh
    public Double[] score(List<String> features) {
        Double[] labelScores = new Double[labelMap.length] ;

        for (int labelIdx=0;labelIdx< labelMap.length; labelIdx++)
        {
            double score=0.;
            HashMap<String, Double> w = avgWeights[labelIdx];
            for (String feat : features) {
                if (w.containsKey(feat))
                    score += w.get(feat);
            }
            labelScores[labelIdx]= score;
        }
        return labelScores;
    }


    private int argmax(List<String> features, HashMap<String,Double> realValuedFeatures, boolean decode) {
        HashMap<String, Double>[] map = decode ? avgWeights : weights;
        double max = Double.NEGATIVE_INFINITY;
        int argmax = 0;

        for (int i = 0; i < map.length; i++) {
            double score = 0.0;

            HashMap<String, Double> w = map[i];
            for (String feat : features) {
                if (w.containsKey(feat))
                    score += w.get(feat);
            }
            for(String feat:realValuedFeatures.keySet()){
                if (w.containsKey(feat))
                    score += w.get(feat)*realValuedFeatures.get(feat);
            }

            if (score > max) {
                max = score;
                argmax = i;
            }
        }

        return argmax;
    }

    public void saveModel(String filePath) throws Exception {
        HashMap<String, Double>[] newAvgMap = new HashMap[labelMap.length];
        for (int i = 0; i < newAvgMap.length; i++) {
            newAvgMap[i] = new HashMap<String, Double>();
            for (String feat : weights[i].keySet()) {
                newAvgMap[i]
                        .put(feat, weights[i].get(feat) - (avgWeights[i].get(feat) / iteration));
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
        HashMap<String, Double>[] newAvgWeight =
                (HashMap<String, Double>[]) reader.readObject();
        String[] labelMap = (String[]) reader.readObject();
        HashMap<String, Integer> reverseLabelMap = (HashMap<String, Integer>) reader.readObject();

        fis.close();
        gz.close();
        reader.close();

        return new AveragedPerceptron(newAvgWeight, labelMap, reverseLabelMap);

    }

}