package ml;

import SupervisedSRL.Strcutures.CompactArray;

import java.io.*;
import java.text.DecimalFormat;
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
    private HashMap<Object, CompactArray>[] weights;
    private HashMap<Object, CompactArray>[] avgWeights;

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
            weights[i] = new HashMap<Object, CompactArray>();
            avgWeights[i] = new HashMap<Object, CompactArray>();
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


    public AveragedPerceptron(HashMap<Object, CompactArray>[] avgWeights, String[] labelMap,
                               HashMap<String, Integer> reverseLabelMap) {
        this.avgWeights = avgWeights;
        this.labelMap = labelMap;
        this.reverseLabelMap = reverseLabelMap;
    }

    public HashMap<Object, CompactArray>[] getWeights() {return weights;}

    public HashMap<Object, CompactArray>[] getAvgWeights() {return avgWeights;}

    public int getIteration() {return iteration;}

    public String[] getLabelMap() {return labelMap;}

    public HashMap<String, Integer> getReverseLabelMap() {return reverseLabelMap;}


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

    private void updateWeight(int argmax, int gold, Object[] features) {
        for (int i=0;i<features.length;i++) {
            updateWeight(argmax,i, features[i], -1);
            updateWeight(gold,i, features[i], 1);
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

        for (int f=0;f<features.length;f++) {
            if (map[f].containsKey(features[f])) {
                CompactArray w = map[f].get(features[f]);
                int offset= w.getOffset();
                for (int i = 0; i < w.length(); i++)
                    score[i+ offset] += w.getArray()[i];
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


    public double[] score(Object[] features) {
        double[] score = new double[labelMap.length];

        for (int f=0;f<features.length;f++) {
            if (avgWeights[f].containsKey(features[f])) {
                CompactArray w = avgWeights[f].get(features[f]);
                int offset= w.getOffset();
                for (int i = 0; i < w.length(); i++)
                    score[i+ offset] += w.getArray()[i];
            }
        }
        return score;
    }

    // todo can copy this to the new class "ModelInformation" and just add writeObject(indexMaps)
    public void saveModel(String filePath) throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        HashMap<Object, CompactArray>[] newAvgMap = new HashMap[weights.length];

        for(int f=0;f<weights.length;f++) {
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


    public AveragedPerceptron calculateAvgWeights ()
    {
        HashMap<Object, CompactArray>[] newAvgMap = new HashMap[weights.length];

        for(int f=0;f<weights.length;f++) {
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