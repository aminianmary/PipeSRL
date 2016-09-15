package ml;

import SupervisedSRL.Reranker.RerankerInstanceItem;
import SupervisedSRL.Reranker.RerankerPool;
import util.IO;

import java.io.FileOutputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.HashMap;
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
    private double[] weights;
    private double[] avgWeights;
    private int iteration;

    public RerankerAveragedPerceptron(int featureSize) {
        this.iteration = 1;
        weights = new double[featureSize];
        avgWeights = new double[featureSize];
    }

    public RerankerAveragedPerceptron(double[] avgWeights) {
        this.avgWeights = avgWeights;
    }

    public static RerankerAveragedPerceptron loadModel(String filePath) throws Exception {
        double[] newAvgWeight = IO.load(filePath);
        return new RerankerAveragedPerceptron(newAvgWeight);
    }

    public double[] getWeights() {
        return weights;
    }

    public double[] getAvgWeights() {
        return avgWeights;
    }

    public int getIteration() {
        return iteration;
    }

    public void learnInstance(RerankerPool pool) throws Exception {
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
                for (int goldFeat : goldFeats[i].keySet()) {
                    weights[goldFeat] += goldFeats[i].get(goldFeat);
                    avgWeights[goldFeat] += iteration * goldFeats[i].get(goldFeat);
                }
            }

            // decrease the weight for argmax
            if (argmaxFeats[i] != null) {
                for (Integer argmaxFeat : argmaxFeats[i].keySet()) {
                    weights[argmaxFeat] -= argmaxFeats[i].get(argmaxFeat);
                    avgWeights[argmaxFeat] -= iteration * argmaxFeats[i].get(argmaxFeat);
                }
            }
        }
    }

    public int argmax(RerankerPool pool, boolean decode) throws Exception {
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

    private double score(RerankerInstanceItem item, boolean decode) throws Exception {
        double score = 0;
        double[] map = decode ? avgWeights : weights;
        HashMap<Integer, Integer>[] features = item.getFeatures();

        for (int i = 0; i < features.length; i++) {
            //todo check if it is correct
            if (features[i] != null) {
                for (Integer feat : features[i].keySet()) {
                    if (feat == null) continue;
                    if (feat >= map.length) throw new Exception("Unknown feature!");
                    score += map[feat] * features[i].get(feat);
                }
            }
        }
        return score;
    }

    public void saveModel(String filePath) throws Exception {
        double[] newAvgMap = new double[weights.length];
        for (int f = 0; f < weights.length; f++)
            newAvgMap[f] = weights[f] - (avgWeights[f] / iteration);
        FileOutputStream fos = new FileOutputStream(filePath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);
        writer.writeObject(newAvgMap);
        writer.close();
    }
}