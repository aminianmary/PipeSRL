package SupervisedSRL;

import Sentence.Argument;
import Sentence.PA;
import Sentence.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.Prediction;
import ml.AveragedPerceptron;
import util.IO;

import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by Maryam Aminian on 5/23/16.
 */
public class Train {

    //this function is used to train stacked ai-ac models
    public String[] train(String trainData,
                          String devData,
                          int numberOfTrainingIterations,
                          String modelDir,
                          int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures, int aiMaxBeamSize, int acMaxBeamSize) throws Exception {
        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
        HashSet<String> argLabels = IO.obtainLabels(trainSentencesInCONLLFormat);

        final IndexMap indexMap = new IndexMap(trainData);

        //training PD module
        PD.train(trainSentencesInCONLLFormat, indexMap, numberOfTrainingIterations, modelDir, numOfPDFeatures);


        //training AI and AC models separately
        String aiModelPath = trainAI(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, indexMap,
                numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfPDFeatures, aiMaxBeamSize);

        String acModelPath = trainAC(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, argLabels, indexMap,
                numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiMaxBeamSize, acMaxBeamSize);

        return new String[]{aiModelPath, acModelPath};
    }


    //this function is used to train the joint ai-ac model
    public String trainJoint(String trainData,
                             String devData,
                             int numberOfTrainingIterations,
                             String modelDir, String outputPrefix, int numOfFeatures, int numOFPDFeaturs,
                             int maxBeamSize)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
        HashSet<String> argLabels = IO.obtainLabels(trainSentencesInCONLLFormat);
        argLabels.add("0");
        final IndexMap indexMap = new IndexMap(trainData);

        //training PD module
        PD.train(trainSentencesInCONLLFormat, indexMap, numberOfTrainingIterations, modelDir, numOFPDFeaturs);

        AveragedPerceptron ap = new AveragedPerceptron(argLabels, numOfFeatures);

        //training averaged perceptron
        long startTime = 0;
        long endTime = 0;
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            startTime = System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            int s = 0;
            for (String sentence : trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance4JointModel(sentence, indexMap, numOfFeatures);
                ArrayList<Object[]> featVectors = (ArrayList<Object[]>) instances[0];
                ArrayList<String> labels = (ArrayList<String>) instances[1];

                for (int d = 0; d < featVectors.size(); d++) {
                    ap.learnInstance(featVectors.get(d), labels.get(d));
                }
                s++;
                if (s % 1000 == 0)
                    System.out.print(s + "...");
            }
            System.out.print(s + "\n");
            endTime = System.currentTimeMillis();
            System.out.println("Total time of this iteration: " + format.format(((endTime - startTime) / 1000.0) / 60.0));

            System.out.println("****** DEV RESULTS ******");
            System.out.println("Making prediction on dev data started...");
            startTime = System.currentTimeMillis();
            //decoding
            boolean decode = true;
            Decoder decoder = new Decoder(ap.calculateAvgWeights(), "joint");
            TreeMap<Integer, Prediction>[] predictions = new TreeMap[devSentencesInCONLLFormat.size()];
            ArrayList<ArrayList<String>> sentencesToWriteOutputFile = new ArrayList<ArrayList<String>>();

            PD.unseenPreds = 0;
            PD.totalPreds = 0;
            for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {

                if (d % 1000 == 0)
                    System.out.println(d + "/" + devSentencesInCONLLFormat.size());

                Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap, decode);
                sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentencesInCONLLFormat.get(d)));
                predictions[d] = decoder.predictJoint(sentence, indexMap, maxBeamSize, numOfFeatures, numOFPDFeaturs, modelDir);
            }

            IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, ap.getLabelMap(), outputPrefix + "_" + iter);

            //evaluation
            Evaluation.evaluate(outputPrefix + "_" + iter, devData, indexMap, ap.getReverseLabelMap());

            endTime = System.currentTimeMillis();
            System.out.println("Total time for decoding on dev data: " + format.format(((endTime - startTime) / 1000.0) / 60.0));

        }

        System.out.print("\nSaving final model (including indexMap)...");
        String modelPath = modelDir + "/joint.model";
        ModelInfo.saveModel(ap, indexMap, modelPath);
        System.out.println("Done!");

        return modelPath;
    }


    private String trainAI(List<String> trainSentencesInCONLLFormat,
                           List<String> devSentencesInCONLLFormat,
                           IndexMap indexMap,
                           int numberOfTrainingIterations,
                           String modelDir, int numOfFeatures, int numOfPDFeatures, int aiMaxBeamSize)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        HashSet<String> labelSet = new HashSet<String>();
        labelSet.add("1");
        labelSet.add("0");

        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfFeatures);

        //training averaged perceptron
        long startTime = 0;
        long endTime = 0;
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            startTime = System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            int negInstances = 0;
            int dataSize = 0;
            int s = 0;
            ap.correct = 0;
            for (String sentence : trainSentencesInCONLLFormat) {

                Object[] instances = obtainTrainInstance4AI(sentence, indexMap, numOfFeatures);
                ArrayList<Object[]> featVectors = (ArrayList<Object[]>) instances[0];
                ArrayList<String> labels = (ArrayList<String>) instances[1];


                ///////////////////////////////////////////////////////////////////
                /////////******** OVER SAMPLING POSITIVE EXAMPLES ******///////////
                ///////////////////////////////////////////////////////////////////
                /*
                Object[] overSampledInstances = overSample(featVectors, labels);
                featVectors =  (ArrayList<Object[]>) overSampledInstances[0];
                labels =  (ArrayList<String>) overSampledInstances[1];
                 */
                ///////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////


                ///////////////////////////////////////////////////////////////////
                /////////******** DOWN SAMPLING NEG EXAMPLES ******///////////
                ///////////////////////////////////////////////////////////////////
                /*
                Object[] downSampledInstances = downSample(featVectors, labels);
                featVectors =  (ArrayList<Object[]>) downSampledInstances[0];
                labels =  (ArrayList<String>) downSampledInstances[1];
                */
                ///////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////


                for (int d = 0; d < featVectors.size(); d++) {
                    ap.learnInstance(featVectors.get(d), labels.get(d));
                    if (labels.get(d).equals("0"))
                        negInstances++;
                    dataSize++;
                }
                s++;
                if (s % 1000 == 0)
                    System.out.print(s + "...");
            }
            System.out.print(s + "\n");

            double ac = 100. * (double) ap.correct / dataSize;

            System.out.println("data size:" + dataSize + " neg_instances: " + negInstances + " accuracy: " + ac);
            endTime = System.currentTimeMillis();
            System.out.println("Total time for this iteration " + format.format(((endTime - startTime) / 1000.0) / 60.0));

            //making prediction over dev sentences
            System.out.println("****** DEV RESULTS ******");
            //System.out.println("Making prediction on dev data started...");
            //instead of loading model from file, we just calculate the average weights
            Decoder argumentDecoder = new Decoder(ap.calculateAvgWeights(), "AI");
            boolean decode = true;

            //ai confusion matrix
            int[][] aiConfusionMatrix = new int[2][2];
            aiConfusionMatrix[0][0] = 0;
            aiConfusionMatrix[0][1] = 0;
            aiConfusionMatrix[1][0] = 0;
            aiConfusionMatrix[1][1] = 0;


            for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {

                //if (d%1000==0)
                //System.out.println(d+"/"+devSentencesInCONLLFormat.size());
                Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap, decode);
                HashMap<Integer, Prediction> prediction = argumentDecoder.predictAI(sentence, indexMap, aiMaxBeamSize, numOfFeatures, modelDir, numOfPDFeatures);

                //we do evaluation for each sentence and update confusion matrix right here
                aiConfusionMatrix = Evaluation.evaluateAI4ThisSentence(sentence, prediction, aiConfusionMatrix);
            }
            Evaluation.computePrecisionRecall(aiConfusionMatrix);
        }

        System.out.println("\nSaving final model...");
        String modelPath = modelDir + "/AI.model";
        ModelInfo.saveModel(ap, indexMap, modelPath);
        System.out.println("Done!");

        return modelPath;
    }


    private String trainAC(List<String> trainSentencesInCONLLFormat,
                           List<String> devSentencesInCONLLFormat,
                           HashSet<String> labelSet, IndexMap indexMap,
                           int numberOfTrainingIterations,
                           String modelDir, int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures, int aiMaxBeamSize, int acMaxBeamSize)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        //building trainJoint instances
        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfACFeatures);

        //training average perceptron
        long startTime = 0;
        long endTime = 0;
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            startTime = System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            int dataSize = 0;
            int s = 0;
            for (String sentence : trainSentencesInCONLLFormat) {
                Object[] instances = obtainTrainInstance4AC(sentence, indexMap, numOfACFeatures);
                s++;
                ArrayList<Object[]> featVectors = (ArrayList<Object[]>) instances[0];
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
            System.out.println("Total time of this iteration: " + format.format(((endTime - startTime) / 1000.0) / 60.0));

            //making prediction over dev sentences
            System.out.println("****** DEV RESULTS ******");
            //System.out.println("Making prediction on dev data started...");

            //instead of loading model from file, we just calculate the average weights
            String aiModelPath = modelDir + "/AI.model";
            Decoder argumentDecoder = new Decoder(AveragedPerceptron.loadModel(aiModelPath), ap.calculateAvgWeights());
            HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(ap.getReverseLabelMap());
            reverseLabelMap.put("0", reverseLabelMap.size());
            boolean decode = true;

            //ac confusion matrix
            HashMap<Integer, int[]> acConfusionMatrix = new HashMap<Integer, int[]>();
            for (int k = 0; k < reverseLabelMap.keySet().size(); k++) {
                int[] acGoldLabels = new int[reverseLabelMap.keySet().size()];
                acConfusionMatrix.put(k, acGoldLabels);
            }

            for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {

                // if (d%1000==0)
                //System.out.println(d+"/"+devSentencesInCONLLFormat.size());
                Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap, decode);
                HashMap<Integer, Prediction> prediction = argumentDecoder.predictAC(sentence, indexMap,
                        acMaxBeamSize, aiMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures, modelDir);

                //we do evaluation for each sentence and update confusion matrix right here
                acConfusionMatrix = Evaluation.evaluateAC4ThisSentence(sentence, prediction, acConfusionMatrix, reverseLabelMap);
            }
            Evaluation.computePrecisionRecall(acConfusionMatrix, reverseLabelMap);
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
     *
     * @param sentencesInCONLLFormat
     * @param state
     * @param indexMap
     * @param numOfFeatures
     * @return
     */
    private Object[] obtainTrainInstances(List<String> sentencesInCONLLFormat, String state,
                                          IndexMap indexMap, int numOfFeatures) {
        ArrayList<Object[]> featVectors = new ArrayList<Object[]>();
        ArrayList<String> labels = new ArrayList<String>();

        int counter = 0;
        boolean decode = false;
        for (String sentenceInCONLLFormat : sentencesInCONLLFormat) {
            counter++;
            if (counter % 1000 == 0)
                System.out.println(counter + "/" + sentencesInCONLLFormat.size());

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


    private Object[] obtainTrainInstance4AI(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) {
        String state = "AI";
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
                Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, wordIdx,
                        sentence, state, numOfFeatures, indexMap);

                String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : "1";
                featVectors.add(featVector);
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }

    private Object[] obtainTrainInstance4AC(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) {
        String state = "AC";
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
            for (Argument arg : currentArgs) {
                int argIdx = arg.getIndex();
                Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, argIdx,
                        sentence, state, numOfFeatures, indexMap);

                String label = arg.getType();
                featVectors.add(featVector);
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }

    //function is used for joint ai-ac training/decoding
    private Object[] obtainTrainInstance4JointModel(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) {
        String state = "joint";
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
                Object[] featVector = FeatureExtractor.extractFeatures(pIdx, pLabel, wordIdx,
                        sentence, state, numOfFeatures, indexMap);

                String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : isArgument(wordIdx, currentArgs);
                featVectors.add(featVector);
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }

    /////////////////////////////////////////////////////////////////////////////
    //////////////////////////////  SUPPORT FUNCTIONS  /////////////////////////
    ////////////////////////////////////////////////////////////////////////////


    private String isArgument(int wordIdx, ArrayList<Argument> currentArgs) {
        for (Argument arg : currentArgs)
            if (arg.getIndex() == wordIdx)
                return arg.getType();
        return "";
    }

    private List<Integer> obtainSampleIndices(ArrayList<String> labels) {
        ArrayList<Integer> posIndices = new ArrayList<Integer>();
        ArrayList<Integer> negIndices = new ArrayList<Integer>();

        for (int k = 0; k < labels.size(); k++) {
            if (labels.get(k).equals("1"))
                posIndices.add(k);
            else
                negIndices.add(k);
        }

        Collections.shuffle(posIndices);
        Collections.shuffle(negIndices);

        //int sampleSize= Math.min(posIndices.size(), negIndices.size());
        int posSampleSize = posIndices.size();
        int negSampleSize = negIndices.size() / 8;
        List<Integer> sampledPosIndices = posIndices.subList(0, posSampleSize);
        List<Integer> sampledNegIndices = negIndices.subList(0, negSampleSize);

        sampledPosIndices.addAll(sampledNegIndices);
        return sampledPosIndices;
    }

    private Object[] sample(ArrayList<List<String>> featVectors,
                            ArrayList<String> labels,
                            List<Integer> sampleIndices) {
        ArrayList<List<String>> sampledFeatVectors = new ArrayList<List<String>>();
        ArrayList<String> sampledLabels = new ArrayList<String>();

        Collections.shuffle(sampleIndices);
        for (int idx : sampleIndices) {
            sampledFeatVectors.add(featVectors.get(idx));
            sampledLabels.add(labels.get(idx));
        }

        return new Object[]{sampledFeatVectors, sampledLabels};
    }


    private Object[] overSample(ArrayList<Object[]> senFeatVecs, ArrayList<String> senLabels) {
        ArrayList<Integer> negIndices = new ArrayList<Integer>();
        ArrayList<Integer> posIndices = new ArrayList<Integer>();

        ArrayList<Object[]> overSampledFeatVecs = senFeatVecs;
        ArrayList<String> overSampledLabels = senLabels;

        for (int idx = 0; idx < senLabels.size(); idx++) {
            if (senLabels.get(idx).equals("0"))
                negIndices.add(idx);
            else
                posIndices.add(idx);
        }
        int numOfSamples = negIndices.size() - posIndices.size();
        for (int k = 0; k < numOfSamples; k++) {
            if (posIndices.size() != 0) {
                int ranIdx = new Random().nextInt(posIndices.size());
                overSampledFeatVecs.add(senFeatVecs.get(posIndices.get(ranIdx)));
                overSampledLabels.add(senLabels.get(posIndices.get(ranIdx)));
            }
        }
        return new Object[]{overSampledFeatVecs, overSampledLabels};
    }


    private Object[] downSample(ArrayList<Object[]> senFeatVecs, ArrayList<String> senLabels) {
        ArrayList<Integer> negIndices = new ArrayList<Integer>();
        ArrayList<Integer> posIndices = new ArrayList<Integer>();

        ArrayList<Object[]> downSampledFeatVecs = new ArrayList<Object[]>();
        ArrayList<String> downSampledLabels = new ArrayList<String>();

        for (int idx = 0; idx < senLabels.size(); idx++) {
            if (senLabels.get(idx).equals("0"))
                negIndices.add(idx);
            else
                posIndices.add(idx);
        }
        int numOfSamples = posIndices.size();
        ArrayList<Integer> downSampledIndices = new ArrayList<Integer>();
        //adding down sampled neg indices
        for (int k = 0; k < numOfSamples; k++) {
            if (negIndices.size() != 0) {
                int ranIdx = new Random().nextInt(negIndices.size());
                downSampledIndices.add(negIndices.get(ranIdx));
            }
        }
        //adding all pos indices
        downSampledIndices.addAll(posIndices);
        Collections.shuffle(downSampledIndices);

        for (int idx : downSampledIndices) {
            downSampledFeatVecs.add(senFeatVecs.get(idx));
            downSampledLabels.add(senLabels.get(idx));
        }

        return new Object[]{downSampledFeatVecs, downSampledLabels};
    }
}
