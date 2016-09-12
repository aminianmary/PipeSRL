package ml;

import SupervisedSRL.Reranker.RerankerInstanceItem;
import SupervisedSRL.Reranker.RerankerPool;
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
public class RerankerAveragedPerceptron implements Serializable {
    public int correct = 0;
    public int[][] confusionMatrix = new int[2][2];
    private HashMap<Integer, CompactArray>[] weights;
    private HashMap<Integer, CompactArray>[] avgWeights;
    private String[] labelMap;
    private HashMap<String, Integer> reverseLabelMap;
    private int iteration;

    public RerankerAveragedPerceptron(HashSet<String> possibleLabels, int featureTemplateSize) {
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

    public RerankerAveragedPerceptron(HashMap<Integer, CompactArray>[] avgWeights, String[] labelMap,
                                      HashMap<String, Integer> reverseLabelMap) {
        this.avgWeights = avgWeights;
        this.labelMap = labelMap;
        this.reverseLabelMap = reverseLabelMap;
    }

    public static RerankerAveragedPerceptron loadModel(String filePath) throws Exception {
        FileInputStream fis = new FileInputStream(filePath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);
        HashMap<Integer, CompactArray>[] newAvgWeight =
                (HashMap<Integer, CompactArray>[]) reader.readObject();
        String[] labelMap = (String[]) reader.readObject();
        HashMap<String, Integer> reverseLabelMap = (HashMap<String, Integer>) reader.readObject();
        fis.close();
        gz.close();
        reader.close();
        return new RerankerAveragedPerceptron(newAvgWeight, labelMap, reverseLabelMap);

    }

    public HashMap<Integer, CompactArray>[] getWeights() {
        return weights;
    }

    public HashMap<Integer, CompactArray>[] getAvgWeights() {
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

    public void learnInstance(RerankerPool pool) {
        int argmax = argmax(pool, false);

        if (argmax != pool.getGoldIndex()) {
            updateWeight(argmax, pool.getGoldIndex(), pool);
        } else correct++;

        iteration++;
    }

    private void updateWeight(int argmax, int gold, RerankerPool pool) {
        HashMap<Integer, Integer>[] argmaxFeats = pool.item(argmax).getFeatures();
        HashMap<Integer, Integer>[] goldFeats = pool.item(gold).getFeatures();
        for (int i = 0; i < argmaxFeats.length; i++) {
            // increase the weight for gold
            if (goldFeats[i] != null) {
                for (Integer goldFeat : goldFeats[i].keySet()) {
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
            if (argmaxFeats[i] != null) {
                for (Integer argmaxFeat : argmaxFeats[i].keySet()) {
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

    private double score(RerankerInstanceItem item, boolean decode) {
        double score = 0;
        HashMap<Integer, CompactArray>[] map = decode ? avgWeights : weights;
        HashMap<Integer, Integer>[] features = item.getFeatures();

        for (int i = 0; i < features.length; i++) {
            //todo check if it is correct
            if (features[i] != null) {
                for (Integer feat : features[i].keySet()) {
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

        HashMap<Integer, CompactArray>[] newAvgMap = new HashMap[weights.length];

        for (int f = 0; f < weights.length; f++) {
            newAvgMap[f] = new HashMap<Integer, CompactArray>();
            for (Integer feat : weights[f].keySet()) {
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

    public RerankerAveragedPerceptron calculateAvgWeights() {
        HashMap<Integer, CompactArray>[] newAvgMap = new HashMap[weights.length];

        for (int f = 0; f < weights.length; f++) {
            newAvgMap[f] = new HashMap<Integer, CompactArray>();
            for (Integer feat : weights[f].keySet()) {
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
        return new RerankerAveragedPerceptron(newAvgMap, labelMap, reverseLabelMap);
    }

}