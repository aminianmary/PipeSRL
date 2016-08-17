package SupervisedSRL;

import Sentence.Argument;
import Sentence.PA;
import Sentence.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.*;
import de.bwaldvogel.liblinear.*;
import ml.AveragedPerceptron;
import ml.Adam;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;
import util.IO;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Maryam Aminian on 5/23/16.
 */
public class Train {

    //this function is used to train stacked ai-ac models
    public String[] train(String trainData,
                          String devData,
                          String clusterFile,
                          int numberOfTrainingIterations,
                          String modelDir,
                          int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures,
                          int aiMaxBeamSize, int acMaxBeamSize, int adamBatchSize, double learningRate,
                          ClassifierType classifierType, boolean greedy) throws Exception {

        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
        HashSet<String> argLabels = IO.obtainLabels(trainSentencesInCONLLFormat);

        final IndexMap indexMap = new IndexMap(trainData, clusterFile);
        String aiModelPath="";
        String acModelPath="";
        String aiMappingDictsPath ="";
        String acMappingDictsPath="";

        //training PD module
        PD.train(trainSentencesInCONLLFormat, indexMap, Pipeline.numOfPDTrainingIterations, modelDir, numOfPDFeatures);

        if (classifierType == ClassifierType.AveragedPerceptron)
        {
            aiModelPath = trainAI(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, indexMap,
                    numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfPDFeatures, aiMaxBeamSize, greedy);
            acModelPath = trainAC(trainSentencesInCONLLFormat, devData, argLabels, indexMap,
                    numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                    aiMaxBeamSize, acMaxBeamSize, greedy);

        }else if (classifierType == ClassifierType.Liblinear) {
            String[] aiModelFeatDicPath = trainLiblinear(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, "AI",
                    indexMap, modelDir, numOfAIFeatures);
            String[] acModelFeatDicPath = trainLiblinear(trainSentencesInCONLLFormat, devSentencesInCONLLFormat , "AC",
                    indexMap, modelDir, numOfACFeatures);
            aiModelPath= aiModelFeatDicPath[0];
            acModelPath = acModelFeatDicPath[0];
            aiMappingDictsPath= aiModelFeatDicPath[1];
            acMappingDictsPath= acModelFeatDicPath[1];
        }
        else if (classifierType == ClassifierType.Adam) {
            String[] aiModelFeatDicPath = trainAdam(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, devData, indexMap,
                    numberOfTrainingIterations, modelDir, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, "AI",
                    adamBatchSize, aiMaxBeamSize, acMaxBeamSize, learningRate, greedy);
            String[] acModelFeatDicPath = trainAdam(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, devData, indexMap,
                    numberOfTrainingIterations, modelDir, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, "AC",
                    adamBatchSize, aiMaxBeamSize, acMaxBeamSize, learningRate, greedy);

            aiModelPath= aiModelFeatDicPath[0];
            acModelPath = acModelFeatDicPath[0];
            aiMappingDictsPath= aiModelFeatDicPath[1];
            acMappingDictsPath= acModelFeatDicPath[1];
        }
        return new String[]{aiModelPath, aiMappingDictsPath, acModelPath, acMappingDictsPath};
    }


    private String[] trainLiblinear(List<String> trainSentencesInCONLLFormat,
                                   List<String> devSentencesInCONLLFormat,
                                   String taskType,
                                   IndexMap indexMap,
                                   String modelDir, int numOfFeatures)
            throws Exception {

        Pair<HashMap<Object, Integer>[], Pair<HashMap<String, Integer>, Pair<Integer, Integer>>> featLabelDicPair =
                constructFeatureDict4LibLinear(trainSentencesInCONLLFormat, indexMap, numOfFeatures, taskType);
        HashMap<Object, Integer>[] featDict = featLabelDicPair.first;
        HashMap<String, Integer> labelDict = featLabelDicPair.second.first;
        int numOfLiblinearFeatures = featLabelDicPair.second.second.first;
        int numOfTrainInstances = featLabelDicPair.second.second.second;
        String trainLiblinearFormatFile = modelDir+"/train_ll_"+ taskType;
        double bias = 1.0;
        writeLiblinearFeats(trainSentencesInCONLLFormat,indexMap,numOfFeatures,featDict,labelDict,taskType,trainLiblinearFormatFile);
        Problem trainProblem = Problem.readFromFile(new File(trainLiblinearFormatFile), bias);
        assert trainProblem.l== numOfTrainInstances;
        assert trainProblem.n == numOfLiblinearFeatures;
        assert trainProblem.y.length==trainProblem.x.length;
        assert trainProblem.y.length==trainProblem.l;

        SolverType solver = SolverType.L2R_LR; // -s 0
        double C = 1.0;    // cost of constraints violation
        double eps = 0.01; // stopping criteria

        Parameter parameter = new Parameter(solver, C, eps);
        Model model = Linear.train(trainProblem, parameter);

        //MAKING PREDICTION ON DEV DATA
        if (devSentencesInCONLLFormat != null) {
            int goldNum = 0;
            int correct = 0;

            String devLiblinearFormatFile = modelDir + "/dev_ll";
            writeLiblinearFeats(devSentencesInCONLLFormat, indexMap, numOfFeatures, featDict, labelDict, taskType, devLiblinearFormatFile);
            Problem devProblem = Problem.readFromFile(new File(devLiblinearFormatFile), bias);
            assert devProblem.l == numOfTrainInstances;
            assert devProblem.n == numOfLiblinearFeatures;
            assert devProblem.y.length == devProblem.x.length;
            assert devProblem.y.length == devProblem.l;

            for (int i = 0; i < devProblem.l; i++) {
                int goldLabel = (int) devProblem.y[i];
                double[] probEstimates = new double[labelDict.size()];
                int prediction = (int) Linear.predictProbability(model, devProblem.x[i], probEstimates);
                if (prediction == goldLabel)
                    correct++;
                goldNum++;
            }
            double acc = 100 * correct / goldNum;
            System.out.println("accuracy for task " + taskType + ": " + acc);
        }

        String modelPath = modelDir+"/"+taskType+"_ll.model";
        String mappingDictsPath = modelDir+"/mappingDicts_ll_"+taskType;
        ModelInfo.saveModel(model, indexMap, featDict, labelDict, modelPath, mappingDictsPath);
        return new String[]{modelPath, mappingDictsPath};
    }



    private String[] trainAdam (List<String> trainSentencesInCONLLFormat,
                              List<String> devSentencesInCONLLFormat, String devData, IndexMap indexMap, int numberOfTrainingIterations,
                              String modelDir, int numOfPDFeatures, int numOfAIFeatures, int numOfACFeatures, String taskType,
                                int batchSize, int aiMaxBeamSize, int acMaxBeamSize, double learningRate, boolean greedy)throws Exception {
        System.out.println("Training for task " + taskType);
        DecimalFormat format = new DecimalFormat("##.00");

        //training averaged perceptron
        long startTime = 0;
        long endTime = 0;

        double bestFScore = 0.0;
        int noImprovement = 0;

        int numOfFeatures = (taskType.equals("AI"))? numOfAIFeatures: numOfACFeatures;

        String modelPath = modelDir+"/"+taskType+"_adam.model";
        String mappingDictsPath = modelDir+"/mappingDicts_adam_"+taskType;

        ArrayList<ArrayList<Integer>> batchFeatures = new ArrayList<ArrayList<Integer>>();
        ArrayList<String> batchLabels = new ArrayList<String>();
        Pair<HashMap<Object, Integer>[], Pair<HashMap<String, Integer>, Pair<Integer, Integer>>> featLabelDicPair =
                constructFeatureDict4LibLinear(trainSentencesInCONLLFormat,indexMap,numOfFeatures,taskType);
        HashMap<Object, Integer>[] featDict = featLabelDicPair.first;
        HashMap<String, Integer> labelDict = featLabelDicPair.second.first;
        int numOfLiblinearFeatures = featLabelDicPair.second.second.first;
        int numOfTrainInstances = featLabelDicPair.second.second.second;
        HashSet<String> labelSet = new HashSet<String>(labelDict.keySet());
        Adam adam = new Adam(labelSet, numOfLiblinearFeatures, learningRate, 0.9, 0.9999, 1e-8);

        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            System.out.println("<><><><><><><><><><><><><><><><><><><> iter: " + (iter + 1));
            startTime = System.currentTimeMillis();
            int negInstances = 0;
            int dataSize = 0;
            int s = 0;
            adam.correct = 0;

            for (String sentence : trainSentencesInCONLLFormat) {
                Object[] instances = null;
                if (taskType.equals("AI")) instances = obtainTrainInstance4AI(sentence, indexMap, numOfFeatures);
                else if (taskType.equals("AC")) instances = obtainTrainInstance4AC(sentence, indexMap, numOfFeatures);
                ArrayList<Object[]> featVectors = (ArrayList<Object[]>) instances[0];
                ArrayList<String> labels = (ArrayList<String>) instances[1];

                for (int i = 0; i < featVectors.size(); i++) {
                    ArrayList<Integer> ar = new ArrayList<Integer>();
                    for (int d = 0; d < featVectors.get(i).length; d++) {
                        ar.add(featDict[d].get(featVectors.get(i)[d]));
                    }
                    batchFeatures.add(ar);
                }

                batchLabels.addAll(labels);

                if (batchLabels.size() > batchSize) {
                    adam.learnInstance(batchFeatures, batchLabels);
                    dataSize += batchLabels.size();
                    batchFeatures = new ArrayList<ArrayList<Integer>>();
                    batchLabels = new ArrayList<String>();
                }

                s++;
                if (s % 1000 == 0)
                    System.out.print(s + "...");
            }
            adam.learnInstance(batchFeatures, batchLabels);
            dataSize += batchLabels.size();
            batchFeatures = new ArrayList<ArrayList<Integer>>();
            batchLabels = new ArrayList<String>();

            System.out.print(s + "\n");
            double ac = 100. * (double) adam.correct / dataSize;
            System.out.println("data size:" + dataSize + " neg_instances: " + negInstances + " accuracy: " + ac);
            endTime = System.currentTimeMillis();
            System.out.println("Total time for this iteration " + format.format(((endTime - startTime) / 1000.0) / 60.0));

            //MAKING PREDICTION ON DEV DATA
            System.out.println("DEV RESULTS");
            if (devSentencesInCONLLFormat != null) {
                boolean decode = true;
                double f1= 0.0;
                if (taskType.equals("AI")) {
                    //ai confusion matrix
                    int[][] aiConfusionMatrix = new int[2][2];
                    aiConfusionMatrix[0][0] = 0;
                    aiConfusionMatrix[0][1] = 0;
                    aiConfusionMatrix[1][0] = 0;
                    aiConfusionMatrix[1][1] = 0;
                    Decoder argumentDecoder = new Decoder(adam, "AI");

                    for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
                        Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap);
                        HashMap<Integer, Prediction> prediction = argumentDecoder.predictAI(sentence, indexMap, aiMaxBeamSize,
                                numOfAIFeatures, modelDir, numOfPDFeatures, featDict, ClassifierType.Adam, greedy);

                        //we do evaluation for each sentence and update confusion matrix right here
                        aiConfusionMatrix = Evaluation.evaluateAI4ThisSentence(sentence, prediction, aiConfusionMatrix);
                    }
                    f1 = Evaluation.computePrecisionRecall(aiConfusionMatrix);

                }else if (taskType.equals("AC")){
                    String aiModelPath = modelDir+"/AI_adam.model";
                    String aiMappingDicPath = modelDir+"/mappingDicts_adam_AI";
                    String outputFile = modelDir +"_"+taskType+"_dev_output_adam_" + iter;

                    ModelInfo aiModelInfo = new ModelInfo(aiModelPath, aiMappingDicPath, ClassifierType.Adam);
                    Adam aiClassifier = aiModelInfo.getClassifierAdam();
                    Decoder argumentDecoder = new Decoder(aiClassifier, adam);
                    Decoder.decode(argumentDecoder, indexMap, devData, adam.getLabelMap(),
                            aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, aiModelInfo.getFeatDict(), featDict, ClassifierType.Adam, greedy);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(adam.getReverseLabelMap());
                    reverseLabelMap.put("0", reverseLabelMap.size());

                    f1 = Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
                }else if (taskType.equalsIgnoreCase("JOINT")){
                    String outputFile = modelDir +"_"+taskType+"_dev_output_adam_" + iter;

                    Decoder.decode(new Decoder(adam,"JOINT"),
                            indexMap, devData, adam.getLabelMap(),
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, featDict, ClassifierType.Adam, greedy);
                    f1 = Evaluation.evaluate(outputFile, devData, indexMap, labelDict);
                }

                if (f1 > bestFScore) {
                    noImprovement = 0;
                    bestFScore = f1;
                    System.out.print("\nSaving final model...");
                    ModelInfo.saveModel(adam, indexMap, featDict, labelDict,modelPath, mappingDictsPath);
                    System.out.println("Done!");
                } else {
                    noImprovement++;
                    if (noImprovement > 10) {
                        System.out.println("\nEarly stopping...");
                        break;
                    }
                }
            }
        }
        return new String[]{modelPath, mappingDictsPath};
    }


    private void writeLiblinearFeats (List<String> trainSentencesInCONLLFormat, IndexMap indexMap, int numOfFeatures,
                                      HashMap<Object, Integer>[] featDict, HashMap<String, Integer> labelDict,
                                      String taskType, String filePath) throws Exception {
        System.out.println("Writing "+ filePath +"...");
        DecimalFormat format = new DecimalFormat("##.00");
        long startTime = System.currentTimeMillis();
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filePath)));
        int numOfSentences2write = 0;
        StringBuilder sentences2write = new StringBuilder();
        //writing train data in Liblinear format
        for (String sentence : trainSentencesInCONLLFormat) {
            numOfSentences2write++;
            Object[] instances = null;
            if (taskType.equals("AI")) instances = obtainTrainInstance4AI(sentence, indexMap, numOfFeatures);
            else if (taskType.equals("AC")) instances = obtainTrainInstance4AC(sentence, indexMap, numOfFeatures);
            else if (taskType.equalsIgnoreCase("joint"))
                instances = obtainTrainInstance4JointModel(sentence, indexMap, numOfFeatures);

            ArrayList<Object[]> featVectors = (ArrayList<Object[]>) instances[0];
            ArrayList<String> labels = (ArrayList<String>) instances[1];

            for (int i = 0; i < featVectors.size(); i++) {
                int label = labelDict.containsKey(labels.get(i)) ? labelDict.get(labels.get(i)) : -1;
                sentences2write.append(label + " ");
                for (int d = 0; d < featVectors.get(i).length; d++) {
                    if (featDict[d].containsKey(featVectors.get(i)[d]))
                        //seen feature value
                        sentences2write.append(featDict[d].get(featVectors.get(i)[d]) + ":1");
                    else
                        //unseen feature value
                        sentences2write.append(featDict[d].get(Pipeline.unseenSymbol) + ":1");
                    if (d != featVectors.get(i).length - 1)
                        sentences2write.append(" ");
                }
                sentences2write.append("\n");
            }

            if (numOfSentences2write%1000 ==0 || numOfSentences2write ==trainSentencesInCONLLFormat.size())
            {
                writer.write(sentences2write.toString());
                sentences2write= new StringBuilder();
            }
        }
        writer.flush();
        writer.close();
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for writing: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));
        System.out.println("Done!");
    }


    private Pair<HashMap<Object, Integer>[], Pair<HashMap<String, Integer>, Pair<Integer, Integer>>>
    constructFeatureDict4LibLinear(List<String> trainSentencesInCONLLFormat,
                                   IndexMap indexMap, int numOfFeatures, String taskType) throws Exception {
        HashMap<Object, Integer>[] featureDic = new HashMap[numOfFeatures];
        HashSet<Object>[] featuresSeen = new HashSet[numOfFeatures];

        for (int i = 0; i < numOfFeatures; i++) {
            featureDic[i] = new HashMap<Object, Integer>();
            featuresSeen[i] = new HashSet<Object>();
        }
        HashMap<String, Integer> labelDic = new HashMap<String, Integer>();

        System.out.print("Extracting mapping dictionary...");
        DecimalFormat format = new DecimalFormat("##.00");
        long startTime = System.currentTimeMillis();
        int numOfTrainInstances = 0;
        for (String sentence : trainSentencesInCONLLFormat) {
            Object[] instances = null;
            if (taskType.equals("AI")) instances = obtainTrainInstance4AI(sentence, indexMap, numOfFeatures);
            else if (taskType.equals("AC")) instances = obtainTrainInstance4AC(sentence, indexMap, numOfFeatures);
            else if (taskType.equalsIgnoreCase("JOINT"))
                instances = obtainTrainInstance4JointModel(sentence, indexMap, numOfFeatures);
            else if (taskType.equals("PD")) throw new Exception("task not supported");

            ArrayList<Object[]> featVectors = (ArrayList<Object[]>) instances[0]; //in the format averaged perceptron supports
            ArrayList<String> labels = (ArrayList<String>) instances[1];

            numOfTrainInstances += labels.size();
            //getting set of all possible values for each slot
            for (int instance = 0; instance < labels.size(); instance++) {
                for (int dim = 0; dim < numOfFeatures; dim++) {
                    featuresSeen[dim].add(featVectors.get(instance)[dim]);
                }
                if (!labelDic.containsKey(labels.get(instance)))
                    labelDic.put(labels.get(instance), labelDic.size());
            }
        }
        //constructing featureDic
        int featureIndex = 1;
        //for each feature slot
        for (int i = 0; i < numOfFeatures; i++) {
            //adding seen feature indices
            for (Object feat : featuresSeen[i]) {
                featureDic[i].put(feat, featureIndex++);
            }
            //unseen feature index
            featureDic[i].put(Pipeline.unseenSymbol, featureIndex++);
            assert !featuresSeen[i].contains(Pipeline.unseenSymbol);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for extraction" + format.format( ((endTime - startTime)/1000.0)/ 60.0));
        System.out.println("Done!");
        return new Pair<HashMap<Object, Integer>[], Pair<HashMap<String, Integer>, Pair<Integer, Integer>>>(featureDic,
                new Pair<HashMap<String, Integer>, Pair<Integer, Integer>>(labelDic, new Pair<Integer, Integer>(featureIndex, numOfTrainInstances)));
    }


    //this function is used to train the joint ai-ac model
    public String trainJoint(String trainData,
                             String devData,
                             String clusterFile,
                             int numberOfTrainingIterations,
                             String modelDir, String outputPrefix, int numOfFeatures, int numOFPDFeaturs,
                             int maxBeamSize, boolean greedy)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
        HashSet<String> argLabels = IO.obtainLabels(trainSentencesInCONLLFormat);
        argLabels.add("0");
        final IndexMap indexMap = new IndexMap(trainData, clusterFile);

        //training PD module
        PD.train(trainSentencesInCONLLFormat, indexMap, Pipeline.numOfPDTrainingIterations, modelDir, numOFPDFeaturs);

        AveragedPerceptron ap = new AveragedPerceptron(argLabels, numOfFeatures);

        //training averaged perceptron
        long startTime = 0;
        long endTime = 0;
        double bestFScore = 0;
        String modelPath = modelDir + "/joint.model";
        int noImprovement = 0;
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

                Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap);
                sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentencesInCONLLFormat.get(d)));
                predictions[d] = decoder.predictJoint(sentence, indexMap, maxBeamSize, numOfFeatures, numOFPDFeaturs, modelDir, null, ClassifierType.AveragedPerceptron, greedy);
            }

            IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, ap.getLabelMap(), outputPrefix + "_" + iter);

            //evaluation
            double f1 = Evaluation.evaluate(outputPrefix + "_" + iter, devData, indexMap, ap.getReverseLabelMap());
            if (f1 > bestFScore) {
                noImprovement = 0;
                bestFScore = f1;
                System.out.print("\nSaving final model (including indexMap)...");
                ModelInfo.saveModel(ap, indexMap, modelPath);
                System.out.println("Done!");
            } else {
                noImprovement++;
                if (noImprovement > 5) {
                    System.out.print("\nEarly Stopping");
                    break;
                }
            }
            endTime = System.currentTimeMillis();
            System.out.println("Total time for decoding on dev data: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
        }
        return modelPath;
    }


    public String[] trainJointLiblinear(String trainData,
                          String devData, String clusterFile,
                          int numberOfTrainingIterations,
                          String modelDir,
                          int numOfFeatures, int numOfPDFeatures) throws Exception {

        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);

        final IndexMap indexMap = new IndexMap(trainData, clusterFile);
        String modelPath="";
        String mappingDictsPath ="";

        PD.train(trainSentencesInCONLLFormat, indexMap, Pipeline.numOfPDTrainingIterations, modelDir, numOfPDFeatures);
        String[] modelFeatDicPath = trainLiblinear(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, "JOINT", indexMap, modelDir, numOfFeatures);
        modelPath= modelFeatDicPath[0];
        mappingDictsPath=  modelFeatDicPath[1];

        return new String[]{modelPath,mappingDictsPath};
    }


    public String[] trainJointAdam(String trainData,
                                        String devData,
                                        String clusterFile,
                                        int numberOfTrainingIterations,
                                        String modelDir,
                                        int numOfFeatures, int numOfPDFeatures, int adamBatchSize, int maxBeamSize, double learningRate, boolean greedy) throws Exception {

        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);

        final IndexMap indexMap = new IndexMap(trainData, clusterFile);
        String modelPath="";
        String mappingDictsPath ="";

        PD.train(trainSentencesInCONLLFormat, indexMap, Pipeline.numOfPDTrainingIterations, modelDir, numOfPDFeatures);
        String[] modelFeatDicPath = trainAdam(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, devData, indexMap,
                numberOfTrainingIterations, modelDir, numOfPDFeatures, numOfFeatures, numOfFeatures, "JOINT", adamBatchSize, maxBeamSize, maxBeamSize, learningRate, greedy);
        modelPath= modelFeatDicPath[0];
        mappingDictsPath=  modelFeatDicPath[1];

        return new String[]{modelPath,mappingDictsPath};
    }


    private String trainAI(List<String> trainSentencesInCONLLFormat,
                           List<String> devSentencesInCONLLFormat,
                           IndexMap indexMap,
                           int numberOfTrainingIterations,
                           String modelDir, int numOfFeatures, int numOfPDFeatures, int aiMaxBeamSize, boolean greedy)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        HashSet<String> labelSet = new HashSet<String>();
        labelSet.add("1");
        labelSet.add("0");

        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfFeatures);

        //training averaged perceptron
        long startTime = 0;
        long endTime = 0;
        double bestFScore = 0;
        String modelPath = modelDir + "/AI.model";
        int noImprovement = 0;
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

                Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap);
                HashMap<Integer, Prediction> prediction = argumentDecoder.predictAI(sentence, indexMap, aiMaxBeamSize,
                        numOfFeatures, modelDir, numOfPDFeatures, null, ClassifierType.AveragedPerceptron, greedy);

                //we do evaluation for each sentence and update confusion matrix right here
                aiConfusionMatrix = Evaluation.evaluateAI4ThisSentence(sentence, prediction, aiConfusionMatrix);
            }
            double f1 = Evaluation.computePrecisionRecall(aiConfusionMatrix);
            if (f1 > bestFScore) {
                noImprovement = 0;
                bestFScore = f1;
                System.out.print("\nSaving the new model...");
                ModelInfo.saveModel(ap, indexMap, modelPath);
                System.out.println("Done!");
            } else {
                noImprovement++;
                if (noImprovement > 5) {
                    System.out.print("\nEarly stopping...");
                    break;
                }
            }
        }

        return modelPath;
    }


    private String trainAC(List<String> trainSentencesInCONLLFormat,
                           String devData,
                           HashSet<String> labelSet, IndexMap indexMap,
                           int numberOfTrainingIterations,
                           String modelDir, int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures,
                           int aiMaxBeamSize, int acMaxBeamSize, boolean greedy)
            throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        //building trainJoint instances
        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfACFeatures);

        //training average perceptron
        long startTime = 0;
        long endTime = 0;
        double bestFScore = 0;
        int noImprovement = 0;
        String modelPath = modelDir + "/AC.model";
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

            System.out.println("****** DEV RESULTS ******");
            //instead of loading model from file, we just calculate the average weights
            String aiModelPath = modelDir + "/AI.model";
            String outputFile = modelDir + "dev_output_" + iter;

            Decoder argumentDecoder = new Decoder(AveragedPerceptron.loadModel(aiModelPath), ap.calculateAvgWeights());
            Decoder.decode(argumentDecoder, indexMap, devData, ap.getLabelMap(),
                    aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                    modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron, greedy);

            HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(ap.getReverseLabelMap());
            reverseLabelMap.put("0", reverseLabelMap.size());

            double f1 = Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
            if (f1 > bestFScore) {
                noImprovement = 0;
                bestFScore = f1;
                System.out.print("\nSaving final model...");
                ModelInfo.saveModel(ap, modelPath);
                System.out.println("Done!");
            } else {
                noImprovement++;
                if (noImprovement > 5) {
                    System.out.print("\nEarly stopping...");
                    break;
                }
            }
        }
        return modelPath;
    }


    private Object[] obtainTrainInstance4AI(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) throws Exception {
        ArrayList<Object[]> featVectors = new ArrayList<Object[]>();
        ArrayList<String> labels = new ArrayList<String>();
        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceWords = sentence.getWords();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            ArrayList<Argument> currentArgs = pa.getArguments();

            for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx,
                        sentence, numOfFeatures, indexMap);

                String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : "1";
                featVectors.add(featVector);
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }


    private Object[] obtainTrainInstance4AC(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) throws Exception {
        ArrayList<Object[]> featVectors = new ArrayList<Object[]>();
        ArrayList<String> labels = new ArrayList<String>();
        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            ArrayList<Argument> currentArgs = pa.getArguments();
            //extract features for arguments (not all words)
            for (Argument arg : currentArgs) {
                int argIdx = arg.getIndex();
                Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, argIdx, sentence, numOfFeatures, indexMap);

                String label = arg.getType();
                featVectors.add(featVector);
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }


    //function is used for joint ai-ac training/decoding
    private Object[] obtainTrainInstance4JointModel(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) throws Exception {
        ArrayList<Object[]> featVectors = new ArrayList<Object[]>();
        ArrayList<String> labels = new ArrayList<String>();
        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceWords = sentence.getWords();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            ArrayList<Argument> currentArgs = pa.getArguments();

            for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                Object[] featVector = FeatureExtractor.extractJointFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);

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
