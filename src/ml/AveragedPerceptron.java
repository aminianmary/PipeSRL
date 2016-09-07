package ml;

import SupervisedSRL.Reranker.RerankerInstanceItem;
import SupervisedSRL.Reranker.RerankerPool;
import SupervisedSRL.Strcutures.CompactArray;

import java.io.*;
import java.text.DecimalFormat;
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
            weights[i] = new HashMap<Object, CompactArray>();
            avgWeights[i] = new HashMap<Object, CompactArray>();
        }
        labelMap = new String[possibleLabels.size()];
        reverseLabelMap = new HashMap<String, Integer>();
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
        DecimalFormat format = new DecimalFormat("##.00");


        ///System.out.println("loading model...");
        FileInputStream fis = new FileInputStream(filePath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);

        //System.out.println("loading newAvgWeight...");
        long startTime = System.currentTimeMillis();
        HashMap<Object, CompactArray>[] newAvgWeight =
                (HashMap<Object, CompactArray>[]) reader.readObject();
        long endTime = System.currentTimeMillis();
        //System.out.println("Total time to load newAvgWeight: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        //System.out.println("loading labelMap...");
        startTime = System.currentTimeMillis();
        String[] labelMap = (String[]) reader.readObject();
        endTime = System.currentTimeMillis();
        //  System.out.println("Total time to load labelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0 ));

        //  System.out.println("loading reverseLabelMap...");
        startTime = System.currentTimeMillis();
        HashMap<String, Integer> reverseLabelMap = (HashMap<String, Integer>) reader.readObject();
        endTime = System.currentTimeMillis();
        // System.out.println("Total time to load reverseLabelMap: " + format.format( ((endTime - startTime)/1000.0 )/60.0 ));

        fis.close();
        gz.close();
        reader.close();
        // System.out.println("************ DONE ************");
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

    public void learnInstance(Object[] features, String label) {
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

    public void learnInstance(RerankerPool pool) {
        int argmax = argmax(pool, false);

        if (argmax != pool.getGoldIndex()) {
            updateWeight(argmax, pool.getGoldIndex(), pool);
        } else correct++;

        iteration++;
    }

    private void updateWeight(int argmax, int gold, Object[] features) {
        for (int i = 0; i < features.length; i++) {
            updateWeight(argmax, i, features[i], -1);
            updateWeight(gold, i, features[i], 1);
        }
    }

    private void updateWeight(int argmax, int gold, RerankerPool pool) {
        HashMap<Object, Integer>[] argmaxFeats = pool.item(argmax).getFeatures();
        HashMap<Object, Integer>[] goldFeats = pool.item(gold).getFeatures();
        for (int i = 0; i < argmaxFeats.length; i++) {
            // increase the weight for gold
            if (goldFeats[i] != null) {
                for (Object goldFeat : goldFeats[i].keySet()) {
                    CompactArray array = weights[i].get(goldFeat);
                    CompactArray avgArray = avgWeights[i].get(goldFeat);
                    if (array == null) {
                        array = new CompactArray(0, new double[1]);
                        avgArray = new CompactArray(0, new double[1]);
                    }
                    array.expandArray(0, goldFeats[i].get(goldFeat));
                    avgArray.expandArray(0, iteration * goldFeats[i].get(goldFeat));
                    weights[i].put(goldFeat, array);
                    avgWeights[i].put(goldFeat, avgArray);
                }
            }

            // decrease the weight for argmax
            //todo check if it is correct
            if (argmaxFeats[i] != null) {
                for (Object argmaxFeat : argmaxFeats[i].keySet()) {
                    CompactArray array = weights[i].get(argmaxFeat);
                    CompactArray avgArray = avgWeights[i].get(argmaxFeat);
                    if (array == null) {
                        array = new CompactArray(0, new double[1]);
                        avgArray = new CompactArray(0, new double[1]);
                    }
                    array.expandArray(0, -argmaxFeats[i].get(argmaxFeat));
                    avgArray.expandArray(0, -iteration * argmaxFeats[i].get(argmaxFeat));
                    weights[i].put(argmaxFeat, array);
                    avgWeights[i].put(argmaxFeat, avgArray);
                }
            }
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

    public int argmax(RerankerPool pool, boolean decode) {
        double max = Double.NEGATIVE_INFINITY;
        int argmax = 0;

        for (int i = 0; i < pool.length(); i++) {
            double score = score(pool.item(i), decode);
            if (score > max) {
                argmax = i;
                max = score;
            }
        }
        return argmax;
    }


    public double[] score(Object[] features) {
        double[] score = new double[labelMap.length];

        for (int f = 0; f < features.length; f++) {
            if (avgWeights[f].containsKey(features[f])) {
                CompactArray w = avgWeights[f].get(features[f]);
                int offset = w.getOffset();
                for (int i = 0; i < w.length(); i++)
                    score[i + offset] += w.getArray()[i];
            }
        }
        return score;
    }

    private double score(RerankerInstanceItem item, boolean decode) {
        double score = 0;
        HashMap<Object, CompactArray>[] map = decode ? avgWeights : weights;
        HashMap<Object, Integer>[] features = item.getFeatures();

        for (int i = 0; i < features.length; i++) {
            //todo check if it is correct
            if (features[i] != null) {
                for (Object feat : features[i].keySet()) {
                    if (map[i].containsKey(feat)) {
                        double weight = map[i].get(feat).getArray()[0];
                        score += weight * features[i].get(feat);
                    }
                }
            }
        }
        return score;
    }


    public void saveModel(String filePath) throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        HashMap<Object, CompactArray>[] newAvgMap = new HashMap[weights.length];

        for (int f = 0; f < weights.length; f++) {
            newAvgMap[f] = new HashMap<Object, CompactArray>();
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
        //System.out.println("Saving newAvgMap...");
        long startTime = System.currentTimeMillis();
        writer.writeObject(newAvgMap);
        long endTime = System.currentTimeMillis();
        //System.out.println("Total time to save newAvgWeight: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        //System.out.println("Saving labelMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(labelMap);
        endTime = System.currentTimeMillis();
        //System.out.println("Total time to save labelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));


        // System.out.println("Saving reverseLabelMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(reverseLabelMap);
        endTime = System.currentTimeMillis();
        // System.out.println("Total time to save reverseLabelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        writer.close();
    }

    public AveragedPerceptron calculateAvgWeights() {
        HashMap<Object, CompactArray>[] newAvgMap = new HashMap[weights.length];

        for (int f = 0; f < weights.length; f++) {
            newAvgMap[f] = new HashMap<Object, CompactArray>();
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