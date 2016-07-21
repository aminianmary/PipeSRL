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
                              String modelDir, int numOfPDFeatures,
                              int numOfAINominalFeatures, int numOfAIVerbalFeatures,
                              int numOfACNominalFeatures, int numOfACVerbalFeatures) throws Exception
    {
        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        HashSet<String> argLabels = IO.obtainLabels(trainSentencesInCONLLFormat);

        final IndexMap indexMap = new IndexMap(trainData);

        //////////////////////////////////// training PD module /////////////////////////////
        PD.train(trainSentencesInCONLLFormat, indexMap, numberOfTrainingIterations, numOfPDFeatures, modelDir);

        //////////////////////////////// training AI and AC models separately ///////////////
        String aiModelPath = trainAI(trainSentencesInCONLLFormat, indexMap,
                numberOfTrainingIterations, modelDir, numOfAINominalFeatures, numOfAIVerbalFeatures);

        String acModelPath = trainAC(trainSentencesInCONLLFormat, argLabels, indexMap,
                numberOfTrainingIterations, modelDir, numOfACNominalFeatures, numOfACVerbalFeatures);

        return new String[]{aiModelPath, acModelPath};
    }


    //this function is used to train the joint ai-ac model
    public String trainJoint(String trainData,
                                    String devData,
                                    int numberOfTrainingIterations,
                                    String modelDir, int numOfFeatures,
                                    int numOfPDFeatures,
                                    int maxBeamSize)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
        HashSet<String> argLabels = IO.obtainLabels(trainSentencesInCONLLFormat);
        argLabels.add("0");
        final IndexMap indexMap = new IndexMap(trainData);

        //training PD module
        PD.train(trainSentencesInCONLLFormat, indexMap, numberOfTrainingIterations, numOfFeatures, modelDir);

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
                                 String modelDir, int numOfFeaturesNominal, int numOfFeaturesVerbal)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        HashSet<String> labelSet = new HashSet<String>();
        labelSet.add("1");
        labelSet.add("0");

        AveragedPerceptron nominalAP = new AveragedPerceptron(labelSet, numOfFeaturesNominal);
        AveragedPerceptron verbalAP = new AveragedPerceptron(labelSet, numOfFeaturesVerbal);

        //training averaged perceptrons
        long startTime = 0;
        long endTime =0;
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            int s = 0;
            startTime = System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            for(String sentence: trainSentencesInCONLLFormat) {
                s++;
                Object[] instances = obtainTrainInstance4AI (sentence, indexMap, numOfFeaturesNominal, numOfFeaturesVerbal);
                ArrayList<String[]> nominalFeatVectors = (ArrayList<String[]>) instances[0];
                ArrayList<String> nominalLabels = (ArrayList<String>) instances[1];
                ArrayList<String[]> verbalFeatVectors = (ArrayList<String[]>) instances[2];
                ArrayList<String> verbalLabels = (ArrayList<String>) instances[3];

                for (int d = 0; d < nominalFeatVectors.size(); d++) {
                    nominalAP.learnInstance(nominalFeatVectors.get(d), nominalLabels.get(d));
                }

                for (int d = 0; d < verbalFeatVectors.size(); d++) {
                    verbalAP.learnInstance(verbalFeatVectors.get(d), verbalLabels.get(d));
                }
            }
            if (s % 1000 == 0)
                System.out.print(s + "...");
            System.out.print(s + "\n");

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
        ModelInfo.saveModel(nominalAP, verbalAP, indexMap, modelPath);
        System.out.println("Done!");

        return modelPath;
    }



    private String trainAC(List<String> trainSentencesInCONLLFormat,
                           HashSet<String> labelSet, IndexMap indexMap,
                                   int numberOfTrainingIterations,
                                   String modelDir, int numOfFeaturesNominal, int numOfFeaturesVerbal)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        AveragedPerceptron nominalAP = new AveragedPerceptron(labelSet, numOfFeaturesNominal);
        AveragedPerceptron verbalAP = new AveragedPerceptron(labelSet, numOfFeaturesVerbal);

        //training average perceptron
        long startTime =0;
        long endTime =0;
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {

            startTime = System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            int s = 0;
            for (String sentence : trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance4AC (sentence, indexMap, numOfFeaturesNominal, numOfFeaturesVerbal);
                ArrayList<String[]> nominalFeatVectors = (ArrayList<String[]>) instances[0];
                ArrayList<String> nominalLabels = (ArrayList<String>) instances[1];
                ArrayList<String[]> verbalFeatVectors = (ArrayList<String[]>) instances[2];
                ArrayList<String> verbalLabels = (ArrayList<String>) instances[3];

                for (int d = 0; d < nominalFeatVectors.size(); d++) {
                    nominalAP.learnInstance(nominalFeatVectors.get(d), nominalLabels.get(d));
                }

                for (int d = 0; d < verbalFeatVectors.size(); d++) {
                    verbalAP.learnInstance(verbalFeatVectors.get(d), verbalLabels.get(d));
                }
            }
            if (s % 1000 == 0)
                System.out.print(s + "...");
            System.out.print(s + "\n");
            endTime = System.currentTimeMillis();
            System.out.println("Total time of this iteration: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));
        }

        System.out.print("\nSaving final model...");
        String modelPath = modelDir + "/AC.model";
        ModelInfo.saveModel(nominalAP, verbalAP, modelPath);
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


    private Object[] obtainTrainInstance4AI (String sentenceInCONLLFormat, IndexMap indexMap, int numOfNominalFeatures,
                                             int numOfVerbalFeatures) {
        String state= "AI";
        ArrayList<Object[]> nominalFeatVectors = new ArrayList<Object[]>();
        ArrayList<String> nominalLabels = new ArrayList<String>();
        ArrayList<Object[]> verbalFeatVectors = new ArrayList<Object[]>();
        ArrayList<String> verbalLabels = new ArrayList<String>();

        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap, decode);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceWords = sentence.getWords();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            String pLabel = pa.getPredicateLabel();
            boolean isNominal = isNominal(sentence.getPosTags()[pIdx], indexMap);
            ArrayList<Argument> currentArgs = pa.getArguments();

            for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                if (wordIdx != pIdx) {
                    if (isNominal) {
                        Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, wordIdx,
                                sentence, state, numOfNominalFeatures, indexMap);
                        String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : "1";
                        nominalFeatVectors.add(featVector);
                        nominalLabels.add(label);
                    }
                    else{
                        Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, wordIdx,
                                sentence, state, numOfVerbalFeatures, indexMap);
                        String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : "1";
                        verbalFeatVectors.add(featVector);
                        verbalLabels.add(label);
                    }
                }
            }
        }
        return new Object[]{nominalFeatVectors, nominalLabels, verbalFeatVectors, verbalLabels};
    }



    private Object[] obtainTrainInstance4AC (String sentenceInCONLLFormat, IndexMap indexMap,
                                             int numOfNominalFeatures, int numOfVerbalFeatures) {
        String state= "AI";
        ArrayList<Object[]> nominalFeatVectors = new ArrayList<Object[]>();
        ArrayList<String> nominalLabels = new ArrayList<String>();
        ArrayList<Object[]> verbalFeatVectors = new ArrayList<Object[]>();
        ArrayList<String> verbalLabels = new ArrayList<String>();
        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap, decode);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            String pLabel = pa.getPredicateLabel();
            boolean isNominal = isNominal(sentence.getPosTags()[pIdx], indexMap);
            ArrayList<Argument> currentArgs = pa.getArguments();

            for (Argument arg: currentArgs) {
                int argIdx = arg.getIndex();
                if (isNominal) {
                    Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, argIdx,
                            sentence, state, numOfNominalFeatures, indexMap);
                    String label = arg.getType();
                    nominalFeatVectors.add(featVector);
                    nominalLabels.add(label);
                } else {
                    Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, argIdx,
                            sentence, state, numOfVerbalFeatures, indexMap);
                    String label = arg.getType();
                    verbalFeatVectors.add(featVector);
                    verbalLabels.add(label);
                }
            }
        }

        return new Object[]{nominalFeatVectors, nominalLabels, verbalFeatVectors, verbalLabels};
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


    private static boolean isNominal (int ppos, IndexMap indexMap)
    {
        String[] int2StringMap = indexMap.getInt2stringMap();
        String pos = int2StringMap[ppos];
        if (pos.startsWith("VB"))
            return false;
        else
            return true;
    }


}
