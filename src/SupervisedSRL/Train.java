package SupervisedSRL;

import SentenceStruct.Argument;
import SentenceStruct.PA;
import SentenceStruct.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.Prediction;
import SupervisedSRL.Strcutures.ProjectConstantPrefixes;
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
                             ArrayList<String> devSentencesInCONLLFormat,
                             String pdModelDir, String aiModelPath, String acModelPath,
                             IndexMap indexMap,
                             int numberOfTrainingIterations,
                             int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures,
                             int aiMaxBeamSize, int acMaxBeamSize, boolean isModelBuiltOnEntireTrainData) throws Exception {

        HashSet<String> argLabels = IO.obtainLabels(trainSentencesInCONLLFormat);
        //training PD module
        PD.train(trainSentencesInCONLLFormat, indexMap, numberOfTrainingIterations, pdModelDir, numOfPDFeatures);
        trainAI(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, indexMap, numberOfTrainingIterations,
                pdModelDir, aiModelPath, numOfAIFeatures, numOfPDFeatures, aiMaxBeamSize);
        trainAC(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, argLabels, indexMap, numberOfTrainingIterations,
                pdModelDir, aiModelPath, acModelPath, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                aiMaxBeamSize, acMaxBeamSize, isModelBuiltOnEntireTrainData);
    }

    public static void trainAI(List<String> trainSentencesInCONLLFormat,
                               List<String> devSentencesInCONLLFormat,
                               IndexMap indexMap,
                               int numberOfTrainingIterations,
                               String pdModelDir, String aiModelPath, int numOfFeatures, int numOfPDFeatures, int aiMaxBeamSize)
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
                        numOfFeatures, pdModelDir, numOfPDFeatures);

                //we do evaluation for each sentence and update confusion matrix right here
                aiConfusionMatrix = Evaluation.evaluateAI4ThisSentence(sentence, prediction, aiConfusionMatrix);
            }
            double f1 = Evaluation.computePrecisionRecall(aiConfusionMatrix);
            if (f1 > bestFScore) {
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
                               ArrayList<String> devSentencesInCONLLFormat,
                               HashSet<String> labelSet, IndexMap indexMap,
                               int numberOfTrainingIterations,
                               String pdModelDir, String aiModelPath, String acModelPath, int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures,
                               int aiMaxBeamSize, int acMaxBeamSize, boolean isModelBuiltOnEntireTrainData)
            throws Exception {
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
            String tempOutputFile = ProjectConstantPrefixes.TMP_DIR + "AC_dev_output_" + iter;
            Decoder argumentDecoder = new Decoder(AveragedPerceptron.loadModel(aiModelPath), ap.calculateAvgWeights());
            argumentDecoder.decode(indexMap, devSentencesInCONLLFormat,
                    aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                    pdModelDir, tempOutputFile);

            HashMap<String, Integer> reverseLabelMap = new HashMap<>(ap.getReverseLabelMap());
            reverseLabelMap.put("0", reverseLabelMap.size());

            double f1 = Evaluation.evaluate(tempOutputFile, devSentencesInCONLLFormat, indexMap, reverseLabelMap);
            if (f1 > bestFScore) {
                noImprovement = 0;
                bestFScore = f1;
                System.out.print("\nSaving final model...");
                ModelInfo.saveModel(ap, acModelPath);
                if (isModelBuiltOnEntireTrainData)
                    ModelInfo.saveReverseLabelMap(ap.getReverseLabelMap(), acModelPath + ProjectConstantPrefixes.GLOBAL_REVERSE_LABEL_MAP);
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

    public static Object[] obtainTrainInstance4AI(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) throws Exception {
        ArrayList<Object[]> featVectors = new ArrayList<>();
        ArrayList<String> labels = new ArrayList<>();
        boolean decode = false;
        Sentence sentence = new Sentence(sentenceInCONLLFormat, indexMap);
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceWords = sentence.getWords();

        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            ArrayList<Argument> currentArgs = pa.getArguments();

            for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx,
                        sentence, numOfFeatures, indexMap, false, 0);

                String label = (isArgument(wordIdx, currentArgs).equals("")) ? "0" : "1";
                featVectors.add(featVector);
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }

    public static Object[] obtainTrainInstance4AC(String sentenceInCONLLFormat, IndexMap indexMap, int numOfFeatures) throws Exception {
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
                Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, argIdx, sentence, numOfFeatures, indexMap, false, 0);

                String label = arg.getType();
                featVectors.add(featVector);
                labels.add(label);
            }
        }

        return new Object[]{featVectors, labels};
    }

    /////////////////////////////////////////////////////////////////////////////
    //////////////////////////////  SUPPORT FUNCTIONS  /////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    public static String isArgument(int wordIdx, ArrayList<Argument> currentArgs) {
        for (Argument arg : currentArgs)
            if (arg.getIndex() == wordIdx)
                return arg.getType();
        return "";
    }
}
