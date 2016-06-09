package SupervisedSRL;

import java.util.*;

import Sentence.Sentence;
import Sentence.Predicate;
import Sentence.Argument;
import Sentence.PA;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.Pair;
import com.sun.tools.javac.util.ArrayUtils;
import ml.AveragedPerceptron;

/**
 * Created by Maryam Aminian on 5/23/16.
 */
public class Train {

    public static String trainAI(List<String> trainSentencesInCONLLFormat, int numberOfTrainingIterations, String modelDir)
            throws Exception {
        HashSet<String> labelSet = new HashSet<String>();
        labelSet.add("1");
        labelSet.add("0");

        AveragedPerceptron ap = new AveragedPerceptron(labelSet, 93);

        //training averaged perceptron
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            System.out.print("iteration:" + iter + "...\n");
            int negInstances = 0;
            int dataSize = 0;
            int s = 0;
            for(String sentence: trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance(sentence, "AI", 93);
                ArrayList<String[]> featVectors = (ArrayList<String[]>) instances[0];
                ArrayList<String> labels = (ArrayList<String>) instances[1];

                for (int d = 0; d < featVectors.size(); d++) {
                    ap.learnInstance(featVectors.get(d), labels.get(d));
                    if (labels.get(d).equals("0"))
                        negInstances++;
                    dataSize++;

                }
                s++;
                if(s%1000==0)
                    System.out.print(s+"...");
            }
            System.out.print(s+"\n");

            double ac = 100. * (double) ap.correct /dataSize;
            System.out.println("data size:"+ dataSize +" neg_instances: "+negInstances+" accuracy: " + ac);
            int aiTP =  ap.confusionMatrix[1][1];
            int aiFP =  ap.confusionMatrix[1][0];
            int aiFN =  ap.confusionMatrix[0][1];

            System.out.println("AI Precision: " + (double) aiTP / (aiTP + aiFP));
            System.out.println("AI Recall: " + (double) aiTP / (aiTP + aiFN));
            ap.correct = 0;
            ap.confusionMatrix = new int[2][2];
        }

        System.out.print("\nSaving model...");
        String modelPath = modelDir + "/AI.model";
        ap.saveModel(modelPath);
        System.out.println("Done!");

        return modelPath;
    }


    public static Object[] trainAC(List<String> trainSentencesInCONLLFormat, int numberOfTrainingIterations, String modelDir)
            throws Exception {

        //building train instances


        HashSet<String> labelSet = new HashSet<String>();
        AveragedPerceptron ap = new AveragedPerceptron(labelSet, 93);

        //training average perceptron
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            System.out.print("iteration:" + iter + "...\n");
           int dataSize = 0;
            int s = 0;
            for(String sentence: trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance(sentence, "AC", 93);
                s++;
                ArrayList<String[]> featVectors = (ArrayList<String[]>) instances[0];
                ArrayList<String> labels = (ArrayList<String>) instances[1];
                for (int d = 0; d < featVectors.size(); d++) {
                    ap.learnInstance(featVectors.get(d), labels.get(d));
                    dataSize++;
                    labelSet.add(labels.get(d));

                }
                s++;
                if (s % 1000 == 0)
                    System.out.print(s + "...");
            }
            System.out.print(s + "\n");
            double ac = 100. * (double) ap.correct / dataSize;
            System.out.println("accuracy: " + ac);
            ap.correct = 0;
        }

        System.out.print("\nSaving model...");
        String modelPath = modelDir + "/AC.model";
        ap.saveModel(modelPath);
        System.out.println("Done!");

        return new Object[]{modelPath, labelSet};
    }


    private static Object[] obtainTrainInstances(List<String> sentencesInCONLLFormat, String state, int numOfFeatures) {
        ArrayList<String[]> featVectors = new ArrayList<String[]>();
        ArrayList<String> labels = new ArrayList<String>();

        int counter=0;
        for (String sentenceInCONLLFormat : sentencesInCONLLFormat) {
            counter++;
            if (counter%1000==0)
                System.out.println(counter+"/"+sentencesInCONLLFormat.size());

            Sentence sentence = new Sentence(sentenceInCONLLFormat);
            ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            String[] sentenceWords = sentence.getWords();

            for (PA pa : pas) {
                Predicate currentP = pa.getPredicate();
                ArrayList<Argument> currentArgs = pa.getArguments();

                for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                    if (wordIdx != currentP.getIndex()) {
                        String[] featVector = FeatureExtractor.extractFeatures(currentP, wordIdx,
                                sentence, state, numOfFeatures);
                        String label = "";

                        if (state.equals("AI"))
                            label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : "1";
                        else
                            label = pa.obtainArgumentType(wordIdx);

                        featVectors.add(featVector);
                        labels.add(label);
                    }
                }
            }
        }
        /*
        //due to data imbalance --> try to sample pos and neg examples
        List<Integer> sampleIndices= obtainSampleIndices(labels);
        Object[] objs =sample(featVectors, labels, sampleIndices);

        ArrayList<List<String>> sampledFeatVectors = (ArrayList<List<String>>) objs[0];
        ArrayList<String> sampledLabels = (ArrayList<String>) objs[1];

        featVectors= sampledFeatVectors;
        labels= sampledLabels;
        */
        return new Object[]{featVectors, labels};
    }


    private static Object[] obtainTrainInstance(String sentenceInCONLLFormat, String state, int numOfFeatures) {
        ArrayList<String[]> featVectors = new ArrayList<String[]>();
        ArrayList<String> labels = new ArrayList<String>();

        Sentence sentence = new Sentence(sentenceInCONLLFormat);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        String[] sentenceWords = sentence.getWords();

        for (PA pa : pas) {
            Predicate currentP = pa.getPredicate();
            ArrayList<Argument> currentArgs = pa.getArguments();

            for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                if (wordIdx != currentP.getIndex()) {
                    String[] featVector = FeatureExtractor.extractFeatures(currentP, wordIdx,
                            sentence, state, numOfFeatures);
                    String label = "";

                    if (state.equals("AI"))
                        label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : "1";
                    else
                        label = pa.obtainArgumentType(wordIdx);

                    featVectors.add(featVector);
                    labels.add(label);
                }
            }
        }

        return new Object[]{featVectors, labels};
    }


    private static String isArgument(int wordIdx, ArrayList<Argument> currentArgs) {
        for (Argument arg : currentArgs)
            if (arg.getIndex() == wordIdx)
                return arg.getType();
        return "";
    }

    private static List<Integer> obtainSampleIndices(ArrayList<String> labels)
    {
        ArrayList<Integer> posIndices= new ArrayList<Integer>();
        ArrayList<Integer> negIndices= new ArrayList<Integer>();

        for (int k=0; k< labels.size(); k++)
        {
            if(labels.get(k).equals("1"))
                posIndices.add(k);
            else
                negIndices.add(k);
        }

        Collections.shuffle(posIndices);
        Collections.shuffle(negIndices);

        //int sampleSize= Math.min(posIndices.size(), negIndices.size());
        int posSampleSize= posIndices.size();
        int negSampleSize= negIndices.size()/8;
        List<Integer> sampledPosIndices= posIndices.subList(0, posSampleSize);
        List<Integer> sampledNegIndices= negIndices.subList(0, negSampleSize);

        sampledPosIndices.addAll(sampledNegIndices);
        return sampledPosIndices;
    }


    private static Object[] sample(ArrayList<List<String>> featVectors,
                                                  ArrayList<String> labels,
                                                  List<Integer> sampleIndices)
    {
        ArrayList<List<String>> sampledFeatVectors= new ArrayList<List<String>>();
        ArrayList<String> sampledLabels= new ArrayList<String>();

        Collections.shuffle(sampleIndices);
        for (int idx: sampleIndices)
        {
            sampledFeatVectors.add(featVectors.get(idx));
            sampledLabels.add(labels.get(idx));
        }

        return new Object[]{sampledFeatVectors, sampledLabels};
    }

}
