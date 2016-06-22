package SupervisedSRL.Strcutures;

import com.sun.javafx.sg.prism.NGShape;
import ml.AveragedPerceptron;

import java.io.*;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Created by monadiab on 6/22/16.
 */
public class ModelInfo {

    AveragedPerceptron classifier;
    IndexMap indexMap;

    public ModelInfo(AveragedPerceptron classifier, IndexMap indexMap)
    {
        this.classifier= classifier;
        this.indexMap= indexMap;
    }

    public AveragedPerceptron getClassifier() {return classifier;}


    public void saveModel(String filePath) throws Exception {

        HashMap<Object, double[]>[] weights = classifier.getWeights();
        HashMap<Object, double[]>[] avgWeights = classifier.getAvgWeights();
        String[] labelMap = classifier.getLabelMap();
        HashMap<String, Integer> reverseLabelMap= classifier.getReverseLabelMap();
        int iteration= classifier.getIteration();

        HashMap<Object, double[]>[] newAvgMap = new HashMap[weights.length];

        for(int f=0;f<weights.length;f++) {
            newAvgMap[f] = new HashMap<Object, double[]>();
            for (Object feat : weights[f].keySet()) {
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
        writer.writeObject(indexMap);
        writer.close();
    }

    public ModelInfo (String modelPath) throws Exception {
        FileInputStream fis = new FileInputStream(modelPath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);
        HashMap<Object, double[]>[] newAvgWeight =
                (HashMap<Object, double[]>[]) reader.readObject();
        String[] labelMap = (String[]) reader.readObject();
        HashMap<String, Integer> reverseLabelMap = (HashMap<String, Integer>) reader.readObject();
        IndexMap indexMap =(IndexMap) reader.readObject();

        fis.close();
        gz.close();
        reader.close();

        this.classifier = new AveragedPerceptron(newAvgWeight, labelMap, reverseLabelMap);
        this.indexMap= indexMap;
    }
}
