package SupervisedSRL.Strcutures;

import ml.AveragedPerceptron;

import java.io.*;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import de.bwaldvogel.liblinear.*;
import ml.Adam;

/**
 * Created by Maryam Aminian on 6/22/16.
 */
public class ModelInfo implements Serializable {

    AveragedPerceptron classifier;
    Model classifierLiblinear;
    Adam classifierAdam;
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
        FileInputStream fis = new FileInputStream(modelPath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);
        HashMap<Object, CompactArray>[] newAvgWeight =
                (HashMap<Object, CompactArray>[]) reader.readObject();
        String[] labelMap = (String[]) reader.readObject();
        HashMap<String, Integer> reverseLabelMap = (HashMap<String, Integer>) reader.readObject();
        IndexMap indexMap = (IndexMap) reader.readObject();
        fis.close();
        gz.close();
        reader.close();

        this.classifier = new AveragedPerceptron(newAvgWeight, labelMap, reverseLabelMap);
        this.indexMap = indexMap;

    }


    public ModelInfo(String modelPath, String mappingDictsPath, ClassifierType classifierType) throws Exception {

        if (classifierType == ClassifierType.Liblinear)
            this.classifierLiblinear = Linear.loadModel(new File(modelPath));
        else if (classifierType == ClassifierType.Adam)
            this.classifierAdam = Adam.loadModel(modelPath);

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

    public static void saveModel(Adam classifier, IndexMap indexMap, HashMap<Object, Integer>[] featDict,
                                 HashMap<String, Integer> labelDict, String modelPath, String mappingDictsPath) throws Exception {

        classifier.saveModel(modelPath);
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

    public Adam getClassifierAdam() {return classifierAdam;}
}
