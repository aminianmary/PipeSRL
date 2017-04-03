package SupervisedSRL;

import SentenceStruct.Argument;
import SentenceStruct.PA;
import SentenceStruct.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import SentenceStruct.simplePA;
import SupervisedSRL.Strcutures.Pair;
import SupervisedSRL.Strcutures.ProjectConstants;
import ml.AveragedPerceptron;
import util.IO;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 * Created by Maryam Aminian on 5/23/16.
 */
public class Train {
    public static void train(ArrayList<String> trainSentencesInCONLLFormat,
                             ArrayList<String> devSentencesInCONLLFormat, String piModelPath,
                             String aiModelPath, String acModelPath, IndexMap indexMap,
                             int numberOfAITrainingIterations, int numberOfACTrainingIterations,
                             int numOfPIFeatures, int numOfPDFeatures, int numOfAIFeatures, int numOfACFeatures,
                             int aiMaxBeamSize, int acMaxBeamSize, boolean isModelBuiltOnEntireTrainData,
                             double aiCoefficient, String modelsToBeTrained,
                             String trainPDAutoLabelsPath, String pdModelDir,
                             boolean usePI, boolean supplement, boolean weightedLearning) throws Exception {

        HashSet<String> argLabels = IO.obtainLabels(trainSentencesInCONLLFormat);
        if (modelsToBeTrained.contains("AI")) {
            System.out.print("\n>>>> Training AI >>>>\n");
            trainAI(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, indexMap, numberOfAITrainingIterations,
                    piModelPath, aiModelPath, numOfPIFeatures, numOfPDFeatures, numOfAIFeatures, aiMaxBeamSize,
                    trainPDAutoLabelsPath, pdModelDir, usePI, weightedLearning);
            System.out.print("\nDone!\n");
        }
        if (modelsToBeTrained.contains("AC")){
            System.out.print("\n>>>> Training AC >>>>\n");
            trainAC(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, argLabels, indexMap, numberOfACTrainingIterations,
                piModelPath, aiModelPath, acModelPath, numOfPIFeatures, numOfPDFeatures, numOfAIFeatures, numOfACFeatures,
                aiMaxBeamSize, acMaxBeamSize, isModelBuiltOnEntireTrainData, aiCoefficient, trainPDAutoLabelsPath,
                    pdModelDir, usePI, supplement, weightedLearning);
            System.out.print("\nDone!...\n");

        }
    }

    public static void trainAI(List<String> trainSentencesInCONLLFormat,
                               List<String> devSentencesInCONLLFormat,
                               IndexMap indexMap, int numberOfTrainingIterations, String piModelPath, String aiModelPath,
                               int numOfPIFeatures, int numOfPDFeatures, int numOfAIFeatures,
                               int aiMaxBeamSize, String trainPDAutoLabelsPath, String pdModelDir, boolean usePI, boolean weightedLearning)
            throws Exception {
        HashMap<Integer, String>[] trainPDAutoLabels = IO.load(trainPDAutoLabelsPath);

        DecimalFormat format = new DecimalFormat("##.00");
        HashSet<String> labelSet = new HashSet<String>();
        labelSet.add("1");
        labelSet.add("0");
        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfAIFeatures);

        //training averaged perceptron
        long startTime = 0;
        long endTime = 0;
        double bestFScore = 0;
        int noImprovement = 0;

        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            startTime = System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            int negInstances = 0;
            int dataSize = 0;
            int s = 0;
            ap.correct = 0;

            for (int sID=0; sID< trainSentencesInCONLLFormat.size(); sID++) {
                Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(sID), indexMap);
                int[] depLabels = sentence.getDepLabels();
                int[] sourceDepLabels = sentence.getSourceDepLabels();

                Object[] instances = obtainTrainInstance4AI(sentence, indexMap, numOfAIFeatures,
                        trainPDAutoLabels[sID]);
                ArrayList<Pair> featVectors = (ArrayList<Pair>) instances[0];
                ArrayList<String> labels = (ArrayList<String>) instances[1];
                //double learningWeight = (weightedLearning) ? sentence.getCompletenessDegree() : 1;

                for (int d = 0; d < featVectors.size(); d++) {
                    int wIdx = (int) featVectors.get(d).second;
                    Object[] f = (Object[]) featVectors.get(d).first;
                    double learningWeight = (weightedLearning)? ((depLabels[wIdx]== sourceDepLabels[wIdx])?1:0.5):1;
                    ap.learnInstance(f, labels.get(d), learningWeight);
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

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++)
                    System.out.print(ap.confusionMatrix[i][j] + "\t");
                System.out.println("");
            }
            ap.confusionMatrix = new int[2][2];

            System.out.println("data size:" + dataSize + " neg_instances: " + negInstances + " accuracy: " + ac);
            endTime = System.currentTimeMillis();
            System.out.println("Total time for this iteration " + format.format(((endTime - startTime) / 1000.0) / 60.0));

            //making prediction over dev sentences
            System.out.println("****** DEV RESULTS ******");
            //instead of loading model from file, we just calculate the average weights
            AveragedPerceptron piClassifier = (usePI) ? AveragedPerceptron.loadModel(piModelPath) : null;
            Decoder argumentDecoder = new Decoder(piClassifier, ap.calculateAvgWeights(), "AI");
            //ai confusion matrix
            int[][] aiConfusionMatrix = new int[2][2];
            aiConfusionMatrix[0][0] = 0;
            aiConfusionMatrix[0][1] = 0;
            aiConfusionMatrix[1][0] = 0;
            aiConfusionMatrix[1][1] = 0;

            for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
                Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap);
                HashMap<Integer, simplePA> prediction = argumentDecoder.predictAI(sentence, indexMap, aiMaxBeamSize,
                        numOfPIFeatures, numOfPDFeatures, numOfAIFeatures, pdModelDir, usePI);

                //we do evaluation for each sentence and update confusion matrix right here
                aiConfusionMatrix = Evaluation.evaluateAI4ThisSentence(sentence, prediction, aiConfusionMatrix);
            }
            double f1 = Evaluation.computePrecisionRecall(aiConfusionMatrix);
            if (f1 >= bestFScore) {
                noImprovement = 0;
                bestFScore = f1;
                System.out.print("\nSaving the new model...");
                ModelInfo.saveModel(ap, aiModelPath);
                System.out.println("Done!");
            } else {
                noImprovement++;
                if (noImprovement > 5) {
                    System.out.print("\nEarly stopping...");
                    break;
                }
            }
        }
    }


    public static void trainAC(ArrayList<String> trainSentencesInCONLLFormat,
                               ArrayList<String> devSentencesInCONLLFormat, HashSet<String> labelSet, IndexMap indexMap,
                               int numberOfTrainingIterations, String piModelPath, String aiModelPath, String acModelPath,
                               int numOfPIFeatures, int numOfPDFeatures, int numOfAIFeatures,int numOfACFeatures,
                               int aiMaxBeamSize, int acMaxBeamSize,
                               boolean isModelBuiltOnEntireTrainData, double aiCoefficient,
                               String trainPDAutoLabelsPath, String pdModelDir, boolean usePI, boolean supplement, boolean weightedLearning)
            throws Exception {
        HashMap<Integer, String>[] trainPDAutoLabels = IO.load(trainPDAutoLabelsPath);
        DecimalFormat format = new DecimalFormat("##.00");
        //building trainJoint instances
        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfACFeatures);

        //training average perceptron
        long startTime = 0;
        long endTime = 0;
        double bestFScore = 0;
        int noImprovement = 0;
        for (int iter = 0; iter < numberOfTrainingIterations; iter++) {
            startTime = System.currentTimeMillis();
            System.out.print("iteration:" + iter + "...\n");
            int dataSize = 0;
            int s = 0;

            for (int sID=0; sID< trainSentencesInCONLLFormat.size() ; sID++) {
                Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(sID), indexMap);
                int[] depLabels = sentence.getDepLabels();
                int[] sourceDepLabels = sentence.getSourceDepLabels();

                Object[] instances = obtainTrainInstance4AC(sentence, indexMap, numOfACFeatures, trainPDAutoLabels[sID]);
                s++;
                ArrayList<Pair> featVectors = (ArrayList<Pair>) instances[0];
                ArrayList<String> labels = (ArrayList<String>) instances[1];

                for (int d = 0; d < featVectors.size(); d++) {
                    int wIdx = (int) featVectors.get(d).second;
                    Object[] f = (Object[]) featVectors.get(d).first;
                    //double learningWeight = (weightedLearning)? sentence.getCompletenessDegree():1;
                    double learningWeight = (weightedLearning)? ((depLabels[wIdx]== sourceDepLabels[wIdx])?1:0.5):1;
                    ap.learnInstance(f, labels.get(d), learningWeight);
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
            String tempOutputFile = ProjectConstants.TMP_DIR + "AC_dev_output_" + iter;
            String tempOutputFile_w_projected_info = ProjectConstants.TMP_DIR + "AC_dev_output_w_projected_info_" + iter;

            AveragedPerceptron piClassifier = (usePI) ? AveragedPerceptron.loadModel(piModelPath): null;
            Decoder argumentDecoder = new Decoder(piClassifier, AveragedPerceptron.loadModel(aiModelPath), ap.calculateAvgWeights());

            argumentDecoder.decode(indexMap, devSentencesInCONLLFormat, aiMaxBeamSize, acMaxBeamSize,
                    numOfPIFeatures, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, tempOutputFile,
                    tempOutputFile_w_projected_info,aiCoefficient, pdModelDir, usePI, supplement);

            HashMap<String, Integer> reverseLabelMap = new HashMap<>(ap.getReverseLabelMap());
            reverseLabelMap.put("0", reverseLabelMap.size());

            double f1 = Evaluation.evaluate(tempOutputFile_w_projected_info, devSentencesInCONLLFormat, indexMap, reverseLabelMap);
            if (f1 >= bestFScore) {
                noImprovement = 0;
                bestFScore = f1;
                System.out.print("\nSaving final model...");
                ModelInfo.saveModel(ap, acModelPath);
                if (isModelBuiltOnEntireTrainData)
                    IO.write(ap.getReverseLabelMap(), acModelPath + ProjectConstants.GLOBAL_REVERSE_LABEL_MAP);
                System.out.println("Done!");
            } else {
                noImprovement++;
                if (noImprovement > 5) {
                    System.out.print("\nEarly stopping...");
                    break;
                }
            }
        }
    }


    public static Object[] obtainTrainInstance4AI(Sentence sentence, IndexMap indexMap, int numOfFeatures,
                                                  HashMap<Integer, String> pdAutoLabels) throws Exception {
        ArrayList<Pair> featVectors = new ArrayList<Pair>();
        ArrayList<String> labels = new ArrayList<>();
        sentence.setPDAutoLabels(pdAutoLabels); //set pd auto labels
        ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceWords = sentence.getWords();

        for (PA pa : goldPAs) {
            int goldPIdx = pa.getPredicate().getIndex();
            ArrayList<Argument> goldArgs = pa.getArguments();

            for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                String argLabel = getArgLabel(wordIdx, goldArgs);
                Object[] featVector = FeatureExtractor.extractAIFeatures(goldPIdx, wordIdx,
                        sentence, numOfFeatures, indexMap, false, 0); //sentence object must have pd auto labels now
                String label = (argLabel.equals("")) ? "0" : "1";
                featVectors.add(new Pair(featVector,wordIdx));
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }

    public static Object[] obtainTrainInstance4AC(Sentence sentence, IndexMap indexMap, int numOfFeatures,
                                                  HashMap<Integer, String> pdAutoLabels) throws Exception {
        ArrayList<Pair> featVectors = new ArrayList<Pair>();
        ArrayList<String> labels = new ArrayList<String>();
        sentence.setPDAutoLabels(pdAutoLabels); //set pd auto labels
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();

        for (PA pa : pas) {
            int pIdx = pa.getPredicate().getIndex();
            ArrayList<Argument> currentArgs = pa.getArguments();
            //extract features for arguments (not all words)
            for (Argument arg : currentArgs) {
                int argIdx = arg.getIndex();
                String label = arg.getType();
                //check if this word is undecided or not?!
                Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, argIdx, sentence, numOfFeatures, indexMap, false, 0);
                featVectors.add(new Pair(featVector, argIdx));
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }

    /////////////////////////////////////////////////////////////////////////////
    //////////////////////////////  SUPPORT FUNCTIONS  /////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    public static String getArgLabel(int wordIdx, ArrayList<Argument> currentArgs) {
        for (Argument arg : currentArgs)
            if (arg.getIndex() == wordIdx)
                return arg.getType();
        return "";
    }
}
