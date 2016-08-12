package ml;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/11/16
 * Time: 10:57 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Adam implements Serializable {
    public int correct = 0;
    public int[][] confusionMatrix = new int[2][2];
    double learningRate;
    double beta1;
    double beta2;
    // beta1^t and beta2^t
    double beta1_;
    double beta2_;
    double regularizer;
    Random random;
    double cost;
    double eps;
    // these are for tracking the history
    private double[][] m;
    private double[][] v;
    private double[][] w;
    private double[][] aw;
    private String[] labelMap;
    private HashMap<String, Integer> reverseLabelMap;
    private int t;

    /**
     * @param possibleLabels
     * @param maxNumOfFeatures
     * @param learningRate     good value 0.001
     * @param beta1            good value 0.9
     * @param beta2            good value 0.9999
     */
    public Adam(HashSet<String> possibleLabels, int maxNumOfFeatures, double learningRate, double beta1, double beta2, double regularizer) {
        this.t = 1;
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.beta2 = beta2;
        this.beta1_ = beta1;
        this.beta2_ = beta2;
        this.regularizer = regularizer;
        this.random = new Random();
        double initRange = 0.01;
        this.eps = 1e-8;


        labelMap = new String[possibleLabels.size()];
        reverseLabelMap = new HashMap<String, Integer>();
        int i = 0;
        for (String label : possibleLabels) {
            labelMap[i] = label;
            reverseLabelMap.put(label, i);
            i++;
        }

        System.out.print("initializing the model weights...");

        // one more feature for the bias term
        w = new double[labelMap.length][maxNumOfFeatures + 1];
        aw = new double[labelMap.length][maxNumOfFeatures + 1];
        m = new double[labelMap.length][maxNumOfFeatures + 1];
        v = new double[labelMap.length][maxNumOfFeatures + 1];

        for (i = 0; i < w.length; i++) {
            for (int j = 0; j < w[i].length - 1; j++)
                w[i][j] = random.nextDouble()*2*initRange - initRange;
            w[i][w[i].length - 1] = 0;
        }
        System.out.println("done!");

        confusionMatrix = new int[2][2];
    }


    public Adam(double[][] avgWeights, String[] labelMap,
                HashMap<String, Integer> reverseLabelMap) {
        // put avg weights instead of the original one.
        this.w = avgWeights;
        this.labelMap = labelMap;
        this.reverseLabelMap = reverseLabelMap;
    }

    public static Adam loadModel(String filePath) throws Exception {
        FileInputStream fis = new FileInputStream(filePath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);

        double[][] newAvgWeight = (double[][]) reader.readObject();
        String[] labelMap = (String[]) reader.readObject();
        HashMap<String, Integer> reverseLabelMap = (HashMap<String, Integer>) reader.readObject();

        fis.close();
        gz.close();
        reader.close();
        return new Adam(newAvgWeight, labelMap, reverseLabelMap);
    }

    public String[] getLabelMap() {
        return labelMap;
    }

    public HashMap<String, Integer> getReverseLabelMap() {
        return reverseLabelMap;
    }

    public void learnInstance(ArrayList<ArrayList<Integer>> features, ArrayList<String> labels) throws Exception {
        cost = 0;
        System.out.println("calculating gradients for batch " + labels.size());
        double[][] gradients = calculateGradients(features, labels);
        System.out.println("updating weights");
        updateWeights(gradients);
        System.out.println("regularizing weights");
        regularize();
        System.out.println("averaging weights");
        movingAvg();
        System.out.println("updating iteration");
        incrementT();
        if (Double.isNaN(cost))
            throw new Exception("ERROR! cost NAN");
        System.out.println(getCurrentTimeStamp() + " --> done with this batch --- cost " + cost);
    }

    public String getCurrentTimeStamp() {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date());
    }

    private double[][] calculateGradients(ArrayList<ArrayList<Integer>> features, ArrayList<String> labels) throws Exception {
        int batchSize = labels.size();
        double[][] g = new double[w.length][w[0].length];
        double curCorr = 0;

        for (int i = 0; i < batchSize; i++) {
            ArrayList<Integer> feats = features.get(i);
            int gold = reverseLabelMap.get(labels.get(i));
            double[] probs = new double[labelMap.length];

            int argmax = argmax(feats, probs);
            if (gold == argmax) {
                correct++;
                curCorr++;
            }
            cost -= Math.log(probs[gold]) / batchSize;
            if (Double.isNaN(cost))
                System.out.println("cost NAN error!");

            double[] delta = new double[probs.length];
            for (int l = 0; l < probs.length; l++) {
                int lb = gold == l ? 1 : 0;
                delta[l] = (-lb + probs[l]) / batchSize;
                if (Double.isNaN(delta[l]))
                    throw new Exception("delta is NAN in gradients");

                g[l][g.length - 1] += delta[l];
                for (int f = 0; f < feats.size(); f++) {
                    g[l][f] += delta[l] * w[l][f];
                    if (Double.isNaN(g[l][f]))
                        throw new Exception("g is NAN in gradients");
                }
            }
        }

        System.out.println("batch acc: " + curCorr / batchSize);
        return g;
    }

    private void updateWeights(double[][] g) throws Exception {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g[i][j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * Math.pow(g[i][j], 2);
                if (Double.isNaN(v[i][j]))
                    throw new Exception("v is NAN in update weights");
                if (Double.isNaN(m[i][j]))
                    throw new Exception("m is NAN in update weights");

                double _m = m[i][j] / (1 - beta1_);
                double _v = v[i][j] / (1 - beta2_);

                w[i][j] += -learningRate * _m / (Math.sqrt(_v) + eps);
                if (Double.isNaN(w[i][j]))
                    throw new Exception("m is NAN in update weights");
            }
        }
    }

    private void regularize() throws Exception {
        for (int i = 0; i < w.length; i++) {
            // ignoring bias regularization.
            for (int j = 0; j < w[i].length-1; j++) {
                w[i][j] += 2 * regularizer * w[i][j];
                cost += w[i][j] * w[i][j] * regularizer;
                if (Double.isNaN(cost))
                    throw new Exception("cost is NAN in regularizer");
            }
        }
    }

    private void movingAvg() throws Exception {
        double ratio = Math.min(0.9999, (double) t / (9 + t));
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[i].length; j++) {
                aw[i][j] = (1 - ratio) * w[i][j] + ratio * aw[i][j];
                if (Double.isNaN(aw[i][j]))
                    throw new Exception("cost is NAN in regularizer");
            }
        }
    }

    private void incrementT() {
        beta1_ *= beta1;
        beta2_ *= beta2;
        t++;
    }

    public int argmax(ArrayList<Integer> features, double[] probs) throws Exception {
        int argmax = 0;
        double argmaxValue = Double.NEGATIVE_INFINITY;

        for (int l = 0; l < probs.length; l++) {
            probs[l] += w[l][w[l].length-1];
            for (int f = 0; f < features.size(); f++)
                probs[l] += w[l][features.get(f)];
            if (probs[l] > argmaxValue) {
                argmaxValue = probs[l];
                argmax = l;
            }
        }

        double sum = 0;
        for (int l = 0; l < probs.length; l++) {
            probs[l] = Math.exp(probs[l]-argmaxValue);
            if (Double.isNaN(probs[l]))
                throw new Exception("prob is NAN in regularizer");
            sum += probs[l];
        }

        for (int l = 0; l < probs.length; l++) {
            if (sum != 0)
                probs[l] /= sum;
            else
                probs[l] = 1.0/probs.length;

            if (Double.isNaN(probs[l]))
                throw new Exception("prob is NAN in regularizer");
        }
        return argmax;
    }

    public void saveModel(String filePath) throws Exception {
        FileOutputStream fos = new FileOutputStream(filePath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);
        writer.writeObject(aw);
        writer.writeObject(labelMap);
        writer.writeObject(reverseLabelMap);
        writer.close();
    }
}