package ml;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 5/20/16
 * Time: 1:18 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Test {
    public static void main(String[] args) throws Exception {
        HashSet<String> possibleLabels = new HashSet<String>();
        possibleLabels.add("pos");
        possibleLabels.add("neg");
        int numIter = 3;
        String modelPath = "/tmp/model";

        train(possibleLabels,numIter,modelPath);

        System.out.println("Loading model...");
        AveragedPerceptron classifier = AveragedPerceptron.loadModel(modelPath);
        int correct = 0;
        for(String d:dummyTestData()){
            String prediction = classifier.predict(getFeatures(d));
            if(prediction.equals(getLabel(d)))
                correct++;
            else
                System.out.println(d+"->"+prediction);
        }

        System.out.println(100*(float)correct/dummyTestData().length);

    }

    public static void train(HashSet<String> possibleLabels, int numberOfTrainingIterations,
                             String pathToSaveTheFinalModel) throws  Exception{
        AveragedPerceptron perceptron = new AveragedPerceptron(possibleLabels);

        for(int i=0;i<numberOfTrainingIterations;i++){
            System.out.print("iteration:" + i + "...");
            for(String d:dummyTrainData()){
                perceptron.learnInstance(getFeatures(d), getLabel(d));
            }
        }
        System.out.print("\nSaving model...");
        perceptron.saveModel(pathToSaveTheFinalModel);
        System.out.println("Done!");
    }

    public static String[] dummyTrainData(){
        String[] data = new String[10];
        data[0] = "happy good mood pos";
        data[1] = "happy nice very pos";
        data[2] = "bad awful very neg";
        data[3] = "positive good mood pos";
        data[4] = "happy nice mood pos";
        data[5] = "very hard negative neg";
        data[6] = "happy good mood pos";
        data[7] =  "bad awful very neg";
        data[8] = "good awesome nice pos";
        data[9] =  "disaster awful struggle neg";
        return data;
    }

    public static String[] dummyTestData(){
        String[] data = new String[4];
        data[0] = "happy mood pos";
        data[1] = "nice very pos";
        data[2] = "awful neg";
        data[3] = "positive pos";
        return data;
    }

    public static List<String> getFeatures(String d){
        List<String> feats = new ArrayList<String>();
        String[] spl = d.split(" ");
        for(int i=0;i<spl.length-1;i++)
            feats.add(spl[i]);
        return feats;
    }

    public static String getLabel(String d){
        String[] spl = d.split(" ");
        return  spl[spl.length-1];
    }

}

