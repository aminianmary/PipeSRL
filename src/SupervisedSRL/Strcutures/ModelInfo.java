package SupervisedSRL.Strcutures;

import ml.AveragedPerceptron;

import java.io.*;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import de.bwaldvogel.liblinear.*;

/**
 * Created by Maryam Aminian on 6/22/16.
 */
public class ModelInfo implements Serializable {

    AveragedPerceptron classifier;
    Model classifierLiblinear;
    HashMap<Object, Integer>[] featDict;
    HashMap<String, Integer> labelDict;
    IndexMap indexMap;

    public ModelInfo(AveragedPerceptron classifier, IndexMap indexMap) throws IOException {
        HashMap<Object, CompactArray>[] weights = classifier.getWeights();
        HashMap<Object, CompactArray>[] avgWeights = classifier.getAvgWeights();
        String[] labelMap = classifier.getLabelMap();
        HashMap<String, Integer> reverseLabelMap = classifier.getReverseLabelMap();
        int iteration = classifier.getIteration();

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
        this.classifier = new AveragedPerceptron(newAvgMap, labelMap, reverseLabelMap);
        this.indexMap = indexMap;
    }


    public ModelInfo(String modelPath) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");

        //System.out.println("loading model...");
        FileInputStream fis = new FileInputStream(modelPath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);

        //System.out.println("loading newAvgWeight...");
        long startTime = System.currentTimeMillis();
        HashMap<Object, CompactArray>[] newAvgWeight =
                (HashMap<Object, CompactArray>[]) reader.readObject();
        long endTime = System.currentTimeMillis();
        //System.out.println("Total time to load newAvgWeight: " + format.format( ((endTime - startTime)/1000.0)/60.0 ));

        //System.out.println("loading labelMap...");
        startTime = System.currentTimeMillis();
        String[] labelMap = (String[]) reader.readObject();
        endTime = System.currentTimeMillis();
        // System.out.println("Total time to load labelMap: " + format.format( ((endTime - startTime)/1000.0)/60.0 ));

        // System.out.println("loading reverseLabelMap...");
        startTime = System.currentTimeMillis();
        HashMap<String, Integer> reverseLabelMap = (HashMap<String, Integer>) reader.readObject();
        endTime = System.currentTimeMillis();
        //  System.out.println("Total time to load reverseLabelMap: " + format.format( ((endTime - startTime)/1000.0)/60.0 ));

        // System.out.println("loading indexMap...");
        startTime = System.currentTimeMillis();
        IndexMap indexMap = (IndexMap) reader.readObject();
        endTime = System.currentTimeMillis();
        // System.out.println("Total time to load IndexMap: " + format.format( ((endTime - startTime)/1000.0)/60.0 ));

        fis.close();
        gz.close();
        reader.close();

        this.classifier = new AveragedPerceptron(newAvgWeight, labelMap, reverseLabelMap);
        this.indexMap = indexMap;
        //  System.out.println("************ DONE ************");
    }


    public ModelInfo(String modelPath, String mappingDictsPath) throws Exception {

        this.classifierLiblinear =Linear.loadModel(new File(modelPath));
        FileInputStream fis = new FileInputStream(mappingDictsPath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);
        HashMap<Object, Integer>[] featDict =
                (HashMap<Object, Integer>[]) reader.readObject();
        IndexMap indexMap = (IndexMap) reader.readObject();
        HashMap<String, Integer> labelDict= (HashMap<String, Integer>) reader.readObject();
        fis.close();
        gz.close();
        reader.close();
        this.indexMap = indexMap;
        this.featDict = featDict;
        this.labelDict= labelDict;
    }


    public static void saveModel(AveragedPerceptron classifier, IndexMap indexMap, String filePath) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");
        HashMap<Object, CompactArray>[] weights = classifier.getWeights();
        HashMap<Object, CompactArray>[] avgWeights = classifier.getAvgWeights();
        String[] labelMap = classifier.getLabelMap();
        HashMap<String, Integer> reverseLabelMap = classifier.getReverseLabelMap();
        int iteration = classifier.getIteration();

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


        //System.out.println("Saving reverseLabelMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(reverseLabelMap);
        endTime = System.currentTimeMillis();
        // System.out.println("Total time to save reverseLabelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        //System.out.println("Saving indexMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(indexMap);
        endTime = System.currentTimeMillis();
        //System.out.println("Total time to save indexMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        writer.close();
    }


    public static void saveModel(Model classifier, IndexMap indexMap, HashMap<Object, Integer>[] featDict,
                                 HashMap<String, Integer> labelDict, String modelPath, String mappingDictsPath) throws Exception {

        classifier.save(new File(modelPath));
        //save featDic
        FileOutputStream fos = new FileOutputStream(mappingDictsPath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);
        writer.writeObject(featDict);
        writer.writeObject(indexMap);
        writer.writeObject(labelDict);
        writer.close();
    }


    public static void saveModel(AveragedPerceptron classifier, String filePath) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");

        HashMap<Object, CompactArray>[] weights = classifier.getWeights();
        HashMap<Object, CompactArray>[] avgWeights = classifier.getAvgWeights();
        String[] labelMap = classifier.getLabelMap();
        HashMap<String, Integer> reverseLabelMap = classifier.getReverseLabelMap();
        int iteration = classifier.getIteration();

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
        // System.out.println("Total time to save newAvgWeight: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        // System.out.println("Saving labelMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(labelMap);
        endTime = System.currentTimeMillis();
        // System.out.println("Total time to save labelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));


        // System.out.println("Saving reverseLabelMap...");
        startTime = System.currentTimeMillis();
        writer.writeObject(reverseLabelMap);
        endTime = System.currentTimeMillis();
        // System.out.println("Total time to save reverseLabelMap: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        writer.close();
    }


    public AveragedPerceptron getClassifier() {
        return classifier;
    }


    public IndexMap getIndexMap() {
        return indexMap;
    }

    public Model getClassifierLiblinear() {return classifierLiblinear;}

    public HashMap<Object, Integer>[] getFeatDict() {return featDict;}

    public HashMap<String, Integer> getLabelDict() {return labelDict;}
}
