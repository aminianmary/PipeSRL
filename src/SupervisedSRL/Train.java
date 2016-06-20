package SupervisedSRL;

import java.util.*;

import Sentence.Sentence;
import Sentence.Predicate;
import Sentence.Argument;
import Sentence.PA;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.Pair;
import com.sun.tools.classfile.StackMapTable_attribute;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.util.ArrayUtils;
import ml.AveragedPerceptron;
import sun.awt.AWTAccessor;

/**
 * Created by Maryam Aminian on 5/23/16.
 */
public class Train {

    public static String trainAI(List<String> trainSentencesInCONLLFormat,
                                 List<String> devSentencesInCONLLFormat,
                                 int numberOfTrainingIterations,
                                 String modelDir, int numOfFeatures,
                                 int aiMaxBeamSize)
            throws Exception {
        HashSet<String> labelSet = new HashSet<String>();
        labelSet.add("1");
        labelSet.add("0");

        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfFeatures);

        //training averaged perceptron
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            System.out.print("iteration:" + iter + "...\n");
            int negInstances = 0;
            int dataSize = 0;
            int s = 0;
            for(String sentence: trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance4AI (sentence, numOfFeatures);
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

            //saving the model generated in this iteration for making prediction on dev data
            System.out.print("\nSaving model...");
            String modelPath = modelDir + "/AI.model."+iter;
            ap.saveModel(modelPath);
            System.out.println("Done!");

            System.out.println("****** DEV RESULTS ******");
            //making prediction over dev sentences
            System.out.println("Making prediction on dev data started...");
            ArgumentDecoder argumentDecoder = new ArgumentDecoder(AveragedPerceptron.loadModel(modelDir + "/AI.model."+iter));

            for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {

                if (d%1000==0)
                    System.out.println(d+"/"+devSentencesInCONLLFormat.size());

                Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(d));
                argumentDecoder.predictAI (sentence, aiMaxBeamSize, numOfFeatures);
            }

            argumentDecoder.computePrecisionRecall("AI");
        }

        System.out.print("\nSaving final model...");
        String modelPath = modelDir + "/AI.model";
        ap.saveModel(modelPath);
        System.out.println("Done!");

        return modelPath;
    }

    //this function is used to test if we combine ai and ac modules
    public static String train(List<String> trainSentencesInCONLLFormat,
                                 List<String> devSentencesInCONLLFormat,
                                 int numberOfTrainingIterations,
                                 String modelDir, int numOfFeatures,
                                 HashSet<String> labelSet,
                                int maxBeamSize)
            throws Exception {

        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfFeatures);

        //training averaged perceptron
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            System.out.print("iteration:" + iter + "...\n");
            int negInstances = 0;
            int dataSize = 0;
            int s = 0;
            for(String sentence: trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance (sentence, numOfFeatures);
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

            //saving the model generated in this iteration for making prediction on dev data
            System.out.print("\nSaving model...");
            String modelPath = modelDir + "/AI.model."+iter;
            ap.saveModel(modelPath);
            System.out.println("Done!");

            System.out.println("****** DEV RESULTS ******");
            //making prediction over dev sentences
            System.out.println("Making prediction on dev data started...");
            ArgumentDecoder argumentDecoder = new ArgumentDecoder(AveragedPerceptron.loadModel(modelDir + "/AI.model."+iter));

            for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {

                if (d%1000==0)
                    System.out.println(d+"/"+devSentencesInCONLLFormat.size());

                Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(d));
                argumentDecoder.predictAI (sentence, maxBeamSize, numOfFeatures);
            }

            argumentDecoder.computePrecisionRecall("AI");
        }

        System.out.print("\nSaving final model...");
        String modelPath = modelDir + "/combined.model";
        ap.saveModel(modelPath);
        System.out.println("Done!");

        return modelPath;
    }


    public static String trainAC(List<String> trainSentencesInCONLLFormat, HashSet<String> labelSet,
                                   int numberOfTrainingIterations,
                                   String modelDir, int numOfFeatures)
            throws Exception {

        //building train instances
        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfFeatures);

        //training average perceptron
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            System.out.print("iteration:" + iter + "...\n");
           int dataSize = 0;
            int s = 0;
            for(String sentence: trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance4AC(sentence, numOfFeatures);
                s++;
                ArrayList<String[]> featVectors = (ArrayList<String[]>) instances[0];
                ArrayList<String> labels = (ArrayList<String>) instances[1];
                for (int d = 0; d < featVectors.size(); d++) {
                    ap.learnInstance(featVectors.get(d), labels.get(d));
                    dataSize++;
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

        return modelPath;
    }


    private static Object[] obtainTrainInstances (List<String> sentencesInCONLLFormat, String state, int numOfFeatures) {
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


    private static Object[] obtainTrainInstance4AI (String sentenceInCONLLFormat, int numOfFeatures) {
        String state= "AI";
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

                    String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : "1";
                    featVectors.add(featVector);
                    labels.add(label);
                }
            }
        }

        return new Object[]{featVectors, labels};
    }


    //function is used in tesintg ai-ac modules combination
    private static Object[] obtainTrainInstance (String sentenceInCONLLFormat, int numOfFeatures) {
        String state= "AI";
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

                    String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : isArgument(wordIdx, currentArgs);
                    featVectors.add(featVector);
                    labels.add(label);
                }
            }
        }

        return new Object[]{featVectors, labels};
    }


    private static Object[] obtainTrainInstance4AC (String sentenceInCONLLFormat, int numOfFeatures) {
        String state= "AC";
        ArrayList<String[]> featVectors = new ArrayList<String[]>();
        ArrayList<String> labels = new ArrayList<String>();

        Sentence sentence = new Sentence(sentenceInCONLLFormat);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();

        for (PA pa : pas) {
            Predicate currentP = pa.getPredicate();
            ArrayList<Argument> currentArgs = pa.getArguments();
            //extract features for arguments (not all words)
            for (Argument arg: currentArgs) {
                int argIdx= arg.getIndex();
                String[] featVector = FeatureExtractor.extractFeatures(currentP, argIdx,
                        sentence, state, numOfFeatures);

                String label = arg.getType();
                featVectors.add(featVector);
                labels.add(label);
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

    public static HashSet<String> obtainLabels (List<String> sentences) {
        System.out.println("Getting set of labels...");
        HashSet<String> labels = new HashSet<String>();

        int counter = 0;
        for (String sentence : sentences) {
            counter++;
            if (counter % 1000 == 0)
                System.out.println(counter + "/" + sentences.size());

            String[] tokens = sentence.trim().split("\n");
            for (String token : tokens) {
                String[] fields= token.split("\t");
                for (int k=14 ; k< fields.length; k++)
                    if (!fields[k].equals("_"))
                        labels.add(fields[k]);
            }
        }
        return labels;
    }

}
