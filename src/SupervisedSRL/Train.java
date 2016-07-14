package SupervisedSRL;

import java.text.DecimalFormat;
import java.util.*;

import Sentence.Sentence;
import Sentence.Predicate;
import Sentence.Argument;
import Sentence.PA;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import ml.AveragedPerceptron;
import util.IO;

/**
 * Created by Maryam Aminian on 5/23/16.
 */
public class Train {

    //this function is used to train stacked ai-ac models
    public String[] train (String trainData,
                              int numberOfTrainingIterations,
                              String modelDir,
                              int numOfAIFeatures, int numOfACFeatures) throws Exception
    {
        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        HashSet<String> argLabels = IO.obtainLabels(trainSentencesInCONLLFormat);

        final IndexMap indexMap = new IndexMap(trainData);

        //training PD module
        PD.train(trainSentencesInCONLLFormat, indexMap, numberOfTrainingIterations, modelDir);

        //training AI and AC models separately
        String aiModelPath = trainAI(trainSentencesInCONLLFormat, indexMap,
                numberOfTrainingIterations, modelDir, numOfAIFeatures);

        String acModelPath = trainAC(trainSentencesInCONLLFormat, argLabels, indexMap,
                numberOfTrainingIterations, modelDir, numOfACFeatures);

        return new String[]{aiModelPath, acModelPath};
    }


    //this function is used to train the joint ai-ac model
    public String trainJoint(String trainData,
                                    String devData,
                                    int numberOfTrainingIterations,
                                    String modelDir, int numOfFeatures,
                                    int maxBeamSize)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
        HashSet<String> argLabels = IO.obtainLabels(trainSentencesInCONLLFormat);
        argLabels.add("0");
        final IndexMap indexMap = new IndexMap(trainData);

        //training PD module
        //PD.train(trainSentencesInCONLLFormat, indexMap, numberOfTrainingIterations, modelDir);

        AveragedPerceptron ap = new AveragedPerceptron(argLabels, numOfFeatures);

        //training averaged perceptron
        long startTime = 0;
        long endTime =0;
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            startTime= System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            int negInstances = 0;
            int dataSize = 0;
            int s = 0;
            for(String sentence: trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance4JointModel(sentence, indexMap, numOfFeatures);
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
            endTime = System.currentTimeMillis();
            System.out.println("Total time of this iteration: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

            /*
            System.out.println("****** DEV RESULTS ******");
            System.out.println("Making prediction on dev data started...");
            startTime = System.currentTimeMillis();
            boolean decode = true;
            Decoder decoder = new Decoder(ap.calculateAvgWeights(), "joint");

            for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {

                if (d%1000==0)
                    System.out.println(d+"/"+devSentencesInCONLLFormat.size());

                Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap, decode);
                decoder.predictJoint(sentence, indexMap, maxBeamSize, numOfFeatures, modelDir);
            }

            //decoder.computePrecisionRecall("joint");
            endTime = System.currentTimeMillis();
            System.out.println("Total time for decoding on dev data: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));
            */
        }

        System.out.print("\nSaving final model (including indexMap)...");
        String modelPath = modelDir + "/joint.model";
        ModelInfo.saveModel(ap, indexMap, modelPath);
        System.out.println("Done!");

        return modelPath;
    }


    private String trainAI(List<String> trainSentencesInCONLLFormat,
                                 IndexMap indexMap,
                                 int numberOfTrainingIterations,
                                 String modelDir, int numOfFeatures)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        HashSet<String> labelSet = new HashSet<String>();
        labelSet.add("1");
        labelSet.add("0");

        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfFeatures);

        //training averaged perceptron
        long startTime = 0;
        long endTime =0;
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            startTime = System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            int negInstances = 0;
            int dataSize = 0;
            int s = 0;
            for(String sentence: trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance4AI (sentence, indexMap, numOfFeatures);
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
            endTime = System.currentTimeMillis();
            System.out.println("Total time for this iteration " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

            /*
            //making prediction over dev sentences
            System.out.println("****** DEV RESULTS ******");
            System.out.println("Making prediction on dev data started...");
            //instead of loading model from file, we just calculate the average weights
            ArgumentDecoder argumentDecoder = new ArgumentDecoder(ap.calculateAvgWeights());

            boolean decode = true;
            for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {

                if (d%1000==0)
                    System.out.println(d+"/"+devSentencesInCONLLFormat.size());

                Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap, decode);
                argumentDecoder.predictAI (sentence, indexMap, aiMaxBeamSize, numOfFeatures);
            }
            argumentDecoder.computePrecisionRecall("AI");
            */
        }


        System.out.println("\nSaving final model...");
        String modelPath = modelDir + "/AI.model";
        ModelInfo.saveModel(ap, indexMap, modelPath);
        System.out.println("Done!");

        return modelPath;
    }



    private String trainAC(List<String> trainSentencesInCONLLFormat,
                           HashSet<String> labelSet, IndexMap indexMap,
                                   int numberOfTrainingIterations,
                                   String modelDir, int numOfFeatures)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        //building trainJoint instances
        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfFeatures);

        //training average perceptron
        long startTime =0;
        long endTime =0;
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            startTime = System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            int dataSize = 0;
            int s = 0;
            for (String sentence : trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance4AC(sentence, indexMap, numOfFeatures);
                s++;
                ArrayList<String[]> featVectors = (ArrayList<String[]>) instances[0];
                ArrayList<String> labels = (ArrayList<String>) instances[1];
                for (int d = 0; d < featVectors.size(); d++) {
                    ap.learnInstance(featVectors.get(d), labels.get(d));
                    dataSize++;
                }
                if (s % 1000 == 0)
                    System.out.print(s + "...");
            }
            System.out.print(s + "\n");
            double ac = 100. * (double) ap.correct / dataSize;
            System.out.println("accuracy: " + ac);
            ap.correct = 0;
            endTime = System.currentTimeMillis();
            System.out.println("Total time of this iteration: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));

        }

        System.out.print("\nSaving final model...");
        String modelPath = modelDir + "/AC.model";
        ModelInfo.saveModel(ap, modelPath);
        System.out.println("Done!");

        return modelPath;
    }



    /**
     * This function reads input sentences in Conll format and extracts features vectors
     * This function is no longer used due to high memory usage, instead feature extraction is repeated at each iteration
     * @param sentencesInCONLLFormat
     * @param state
     * @param indexMap
     * @param numOfFeatures
     * @return
     */
    private Object[] obtainTrainInstances (List<String> sentencesInCONLLFormat, String state,
                                                  IndexMap indexMap, int numOfFeatures) {
        ArrayList<Object[]> featVectors = new ArrayList<Object[]>();
        ArrayList<String> labels = new ArrayList<String>();

        int counter=0;
        boolean decode = false;
        for (String sentenceInCONLLFormat : sentencesInCONLLFormat) {
            counter++;
            if (counter%1000==0)
                System.out.println(counter+"/"+sentencesInCONLLFormat.size());

            Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap, decode);
            ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            int[] sentenceWords = sentence.getWords();

            for (PA pa : pas) {
                int pIdx = pa.getPredicateIndex();
                String pLabel = pa.getPredicateLabel();
                ArrayList<Argument> currentArgs = pa.getArguments();

                for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                    if (wordIdx != pIdx) {
                        Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, wordIdx,
                                sentence, state, numOfFeatures, indexMap);
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


    private Object[] obtainTrainInstance4AI (String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) {
        String state= "AI";
        ArrayList<Object[]> featVectors = new ArrayList<Object[]>();
        ArrayList<String> labels = new ArrayList<String>();
        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap, decode);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceWords = sentence.getWords();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            String pLabel = pa.getPredicateLabel();
            ArrayList<Argument> currentArgs = pa.getArguments();

            for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                if (wordIdx != pIdx) {
                    Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, wordIdx,
                            sentence, state, numOfFeatures, indexMap);

                    String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : "1";
                    featVectors.add(featVector);
                    labels.add(label);
                }
            }
        }

        return new Object[]{featVectors, labels};
    }



    private Object[] obtainTrainInstance4AC (String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) {
        String state= "AC";
        ArrayList<Object[]> featVectors = new ArrayList<Object[]>();
        ArrayList<String> labels = new ArrayList<String>();
        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap, decode);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            String pLabel = pa.getPredicateLabel();
            ArrayList<Argument> currentArgs = pa.getArguments();
            //extract features for arguments (not all words)
            for (Argument arg: currentArgs) {
                int argIdx= arg.getIndex();
                Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, argIdx,
                        sentence, state, numOfFeatures, indexMap);

                String label = arg.getType();
                featVectors.add(featVector);
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }



    //function is used in testing ai-ac modules combination
    private Object[] obtainTrainInstance4JointModel(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) {
        String state= "joint";
        ArrayList<Object[]> featVectors = new ArrayList<Object[]>();
        ArrayList<String> labels = new ArrayList<String>();
        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap, decode);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceWords = sentence.getWords();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            String pLabel = pa.getPredicateLabel();
            ArrayList<Argument> currentArgs = pa.getArguments();

            for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                if (wordIdx != pIdx) {
                    Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, wordIdx,
                            sentence, state, numOfFeatures, indexMap);

                    String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : isArgument(wordIdx, currentArgs);
                    featVectors.add(featVector);
                    labels.add(label);
                }
            }
        }

        return new Object[]{featVectors, labels};
    }



    /////////////////////////////////////////////////////////////////////////////
    //////////////////////////////  SUPPORT FUNCTIONS  /////////////////////////
    ////////////////////////////////////////////////////////////////////////////


    private  String isArgument(int wordIdx, ArrayList<Argument> currentArgs) {
        for (Argument arg : currentArgs)
            if (arg.getIndex() == wordIdx)
                return arg.getType();
        return "";
    }

    private  List<Integer> obtainSampleIndices(ArrayList<String> labels)
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


    private  Object[] sample(ArrayList<List<String>> featVectors,
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
