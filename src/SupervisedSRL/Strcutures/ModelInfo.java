package SupervisedSRL.Strcutures;

import com.sun.javafx.sg.prism.NGShape;
import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;
import ml.AveragedPerceptron;

import java.io.*;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Maryam Aminian on 6/22/16.
 */
public class ModelInfo implements Serializable {

    AveragedPerceptron nominalClassifier;
    AveragedPerceptron verbalClassifier;
    IndexMap indexMap;

    public AveragedPerceptron getNominalClassifier() {return nominalClassifier;}


    public AveragedPerceptron getVerbalClassifier() {return verbalClassifier;}


    public IndexMap getIndexMap() {return indexMap;}


    public static void saveModel(AveragedPerceptron classifier, IndexMap indexMap, String filePath) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");
        HashMap<Object, CompactArray>[] weights = classifier.getWeights();
        HashMap<Object, CompactArray>[] avgWeights = classifier.getAvgWeights();
        String[] labelMap = classifier.getLabelMap();
        HashMap<String, Integer> reverseLabelMap= classifier.getReverseLabelMap();
        int iteration= classifier.getIteration();

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

        System.out.println("Saving newAvgMap...");
        long startTime = System.currentTimeMillis();
        writer.writeObject(newAvgMap);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time to save newAvgWeight: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        System.out.println("Saving labelMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(labelMap);
        endTime = System.currentTimeMillis();
        System.out.println("Total time to save labelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));


        System.out.println("Saving reverseLabelMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(reverseLabelMap);
        endTime = System.currentTimeMillis();
        System.out.println("Total time to save reverseLabelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        System.out.println("Saving indexMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(indexMap);
        endTime = System.currentTimeMillis();
        System.out.println("Total time to save indexMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        writer.close();
    }


    public static void saveModel(AveragedPerceptron classifier1, AveragedPerceptron classifier2,
                                 IndexMap indexMap, String filePath) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");
        //getting first classifier info
        HashMap<Object, CompactArray>[] weights1 = classifier1.getWeights();
        HashMap<Object, CompactArray>[] avgWeights1 = classifier1.getAvgWeights();
        String[] labelMap1 = classifier1.getLabelMap();
        HashMap<String, Integer> reverseLabelMap1= classifier1.getReverseLabelMap();
        int iteration1= classifier1.getIteration();

        HashMap<Object, CompactArray>[] newAvgMap1 = new HashMap[weights1.length];

        for (int f = 0; f < weights1.length; f++) {
            newAvgMap1[f] = new HashMap<Object, CompactArray>();
            for (Object feat : weights1[f].keySet()) {
                double[] w = weights1[f].get(feat).getArray();
                double[] aw = avgWeights1[f].get(feat).getArray();
                double[] naw = new double[w.length];
                for (int i = 0; i < w.length; i++) {
                    naw[i] = w[i] - (aw[i] / iteration1);
                }
                CompactArray nawCompact = new CompactArray(weights1[f].get(feat).getOffset(), naw);
                newAvgMap1[f].put(feat, nawCompact);
            }
        }
        //getting second classifier info
        HashMap<Object, CompactArray>[] weights2 = classifier2.getWeights();
        HashMap<Object, CompactArray>[] avgWeights2 = classifier2.getAvgWeights();
        String[] labelMap2 = classifier2.getLabelMap();
        HashMap<String, Integer> reverseLabelMap2= classifier2.getReverseLabelMap();
        int iteration2= classifier2.getIteration();

        HashMap<Object, CompactArray>[] newAvgMap2 = new HashMap[weights1.length];

        for (int f = 0; f < weights1.length; f++) {
            newAvgMap2[f] = new HashMap<Object, CompactArray>();
            for (Object feat : weights1[f].keySet()) {
                double[] w = weights1[f].get(feat).getArray();
                double[] aw = avgWeights1[f].get(feat).getArray();
                double[] naw = new double[w.length];
                for (int i = 0; i < w.length; i++) {
                    naw[i] = w[i] - (aw[i] / iteration1);
                }
                CompactArray nawCompact = new CompactArray(weights1[f].get(feat).getOffset(), naw);
                newAvgMap2[f].put(feat, nawCompact);
            }
        }


        FileOutputStream fos = new FileOutputStream(filePath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);

        System.out.println("Saving newAvgMaps...");
        long startTime = System.currentTimeMillis();
        writer.writeObject(newAvgMap1);
        writer.writeObject(newAvgMap2);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time to save newAvgWeight: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        System.out.println("Saving labelMaps...");
        startTime = System.currentTimeMillis();
        writer.writeObject(labelMap1);
        writer.writeObject(labelMap2);
        endTime = System.currentTimeMillis();
        System.out.println("Total time to save labelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));


        System.out.println("Saving reverseLabelMaps...");
        startTime = System.currentTimeMillis();
        writer.writeObject(reverseLabelMap1);
        writer.writeObject(reverseLabelMap2);
        endTime = System.currentTimeMillis();
        System.out.println("Total time to save reverseLabelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        //saving index map
        System.out.println("Saving indexMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(indexMap);
        endTime = System.currentTimeMillis();
        System.out.println("Total time to save indexMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        writer.close();
    }


    public static void saveModel(AveragedPerceptron classifier, String filePath) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");

        HashMap<Object, CompactArray>[] weights = classifier.getWeights();
        HashMap<Object, CompactArray>[] avgWeights = classifier.getAvgWeights();
        String[] labelMap = classifier.getLabelMap();
        HashMap<String, Integer> reverseLabelMap= classifier.getReverseLabelMap();
        int iteration= classifier.getIteration();

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

        System.out.println("Saving newAvgMap...");
        long startTime = System.currentTimeMillis();
        writer.writeObject(newAvgMap);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time to save newAvgWeight: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        System.out.println("Saving labelMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(labelMap);
        endTime = System.currentTimeMillis();
        System.out.println("Total time to save labelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));


        System.out.println("Saving reverseLabelMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(reverseLabelMap);
        endTime = System.currentTimeMillis();
        System.out.println("Total time to save reverseLabelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        writer.close();
    }


    public static void saveModel(AveragedPerceptron classifier1, AveragedPerceptron classifier2,
                                 String filePath) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");
        //getting first classifier info
        HashMap<Object, CompactArray>[] weights1 = classifier1.getWeights();
        HashMap<Object, CompactArray>[] avgWeights1 = classifier1.getAvgWeights();
        String[] labelMap1 = classifier1.getLabelMap();
        HashMap<String, Integer> reverseLabelMap1= classifier1.getReverseLabelMap();
        int iteration1= classifier1.getIteration();

        HashMap<Object, CompactArray>[] newAvgMap1 = new HashMap[weights1.length];

        for (int f = 0; f < weights1.length; f++) {
            newAvgMap1[f] = new HashMap<Object, CompactArray>();
            for (Object feat : weights1[f].keySet()) {
                double[] w = weights1[f].get(feat).getArray();
                double[] aw = avgWeights1[f].get(feat).getArray();
                double[] naw = new double[w.length];
                for (int i = 0; i < w.length; i++) {
                    naw[i] = w[i] - (aw[i] / iteration1);
                }
                CompactArray nawCompact = new CompactArray(weights1[f].get(feat).getOffset(), naw);
                newAvgMap1[f].put(feat, nawCompact);
            }
        }
        //getting second classifier info
        HashMap<Object, CompactArray>[] weights2 = classifier2.getWeights();
        HashMap<Object, CompactArray>[] avgWeights2 = classifier2.getAvgWeights();
        String[] labelMap2 = classifier2.getLabelMap();
        HashMap<String, Integer> reverseLabelMap2= classifier2.getReverseLabelMap();
        int iteration2= classifier2.getIteration();

        HashMap<Object, CompactArray>[] newAvgMap2 = new HashMap[weights2.length];

        for (int f = 0; f < weights2.length; f++) {
            newAvgMap2[f] = new HashMap<Object, CompactArray>();
            for (Object feat : weights2[f].keySet()) {
                double[] w = weights2[f].get(feat).getArray();
                double[] aw = avgWeights2[f].get(feat).getArray();
                double[] naw = new double[w.length];
                for (int i = 0; i < w.length; i++) {
                    naw[i] = w[i] - (aw[i] / iteration2);
                }
                CompactArray nawCompact = new CompactArray(weights2[f].get(feat).getOffset(), naw);
                newAvgMap2[f].put(feat, nawCompact);
            }
        }

        FileOutputStream fos = new FileOutputStream(filePath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);

        System.out.println("Saving newAvgMaps...");
        long startTime = System.currentTimeMillis();
        writer.writeObject(newAvgMap1);
        writer.writeObject(newAvgMap2);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time to save newAvgWeight: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        System.out.println("Saving labelMaps...");
        startTime = System.currentTimeMillis();
        writer.writeObject(labelMap1);
        writer.writeObject(labelMap2);
        endTime = System.currentTimeMillis();
        System.out.println("Total time to save labelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));


        System.out.println("Saving reverseLabelMaps...");
        startTime = System.currentTimeMillis();
        writer.writeObject(reverseLabelMap1);
        writer.writeObject(reverseLabelMap2);
        endTime = System.currentTimeMillis();
        System.out.println("Total time to save reverseLabelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        writer.close();
    }


    public ModelInfo (String modelPath, boolean containsIndexMap) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");

        System.out.println("loading model...");
        FileInputStream fis = new FileInputStream(modelPath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);

        System.out.println("loading newAvgWeights...");
        long startTime = System.currentTimeMillis();
        HashMap<Object, CompactArray>[] newAvgWeight1 =
                (HashMap<Object, CompactArray>[]) reader.readObject();
        HashMap<Object, CompactArray>[] newAvgWeight2 =
                (HashMap<Object, CompactArray>[]) reader.readObject();
        long endTime = System.currentTimeMillis();
        System.out.println("Total time to load newAvgWeight: " + format.format( ((endTime - startTime)/1000.0)/60.0 ));

        System.out.println("loading labelMaps...");
        startTime = System.currentTimeMillis();
        String[] labelMap1 = (String[]) reader.readObject();
        String[] labelMap2 = (String[]) reader.readObject();
        endTime = System.currentTimeMillis();
        System.out.println("Total time to load labelMap: " + format.format( ((endTime - startTime)/1000.0)/60.0 ));

        System.out.println("loading reverseLabelMaps...");
        startTime = System.currentTimeMillis();
        HashMap<String, Integer> reverseLabelMap1 = (HashMap<String, Integer>) reader.readObject();
        HashMap<String, Integer> reverseLabelMap2 = (HashMap<String, Integer>) reader.readObject();
        endTime = System.currentTimeMillis();
        System.out.println("Total time to load reverseLabelMap: " + format.format( ((endTime - startTime)/1000.0)/60.0 ));

        if (containsIndexMap) {
            System.out.println("loading indexMap...");
            startTime = System.currentTimeMillis();
            IndexMap indexMap = (IndexMap) reader.readObject();
            endTime = System.currentTimeMillis();
            System.out.println("Total time to load IndexMap: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
            this.indexMap= indexMap;
        }

        fis.close();
        gz.close();
        reader.close();

        this.nominalClassifier = new AveragedPerceptron(newAvgWeight1, labelMap1, reverseLabelMap1);
        this.verbalClassifier = new AveragedPerceptron(newAvgWeight2, labelMap2, reverseLabelMap2);
        System.out.println("************ DONE ************");
    }
}
