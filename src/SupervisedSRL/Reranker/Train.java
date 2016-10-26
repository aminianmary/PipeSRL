package SupervisedSRL.Reranker;

import SupervisedSRL.Evaluation;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import ml.AveragedPerceptron;
import ml.RerankerAveragedPerceptron;
import util.IO;

import java.io.EOFException;
import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;

/**
 * Created by Maryam Aminian on 8/25/16.
 */
public class Train {
    public static void trainReranker(Properties properties) throws Exception {
        int numOfPartitions = properties.getNumOfPartitions();
        int numOfTrainingIterations = properties.getMaxNumOfRerankerTrainingIterations();
        String rerankerModelPath = properties.getRerankerModelPath();
        double aiCoefficient = properties.getAiCoefficient();

        RerankerAveragedPerceptron ap = new RerankerAveragedPerceptron(numOfFeatures(properties));
        double bestFScore = 0;
        int noImprovement = 0;

        for (int iter = 0; iter < numOfTrainingIterations; iter++) {
            System.out.println("Iteration " + iter + "\n>>>>>>>>>>>\n");

            for (int devPart = 0; devPart < numOfPartitions; devPart++) {
                System.out.println("Loading/learning train instances for dev part " + devPart + "\n");
                FileInputStream fis = new FileInputStream(properties.getRerankerInstancesFilePath(devPart));
                GZIPInputStream gz = new GZIPInputStream(fis);
                ObjectInput reader = new ObjectInputStream(gz);

                while (true) {
                    try {
                        RerankerPool pool = (RerankerPool) reader.readObject();
                        ap.learnInstance(pool);
                    } catch (EOFException e) {
                        reader.close();
                        break;
                    }
                }
                System.out.println("Part " + devPart + " done!");
            }

            System.out.print("Making prediction on Dev data...");
            HashMap<Object, Integer>[] rerankerFeatureMap = IO.load(properties.getRerankerFeatureMapPath());
            AveragedPerceptron piClassifier = AveragedPerceptron.loadModel(properties.getPiModelPath());
            AveragedPerceptron aiClassifier = AveragedPerceptron.loadModel(properties.getAiModelPath());
            AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(properties.getAcModelPath());
            IndexMap indexMap = IO.load(properties.getIndexMapFilePath());
            ArrayList<String> devSentences = IO.readCoNLLFile(properties.getDevFile());
            int numOfPIFeatures = properties.getNumOfPIFeatures();
            int numOfPDFeatures = properties.getNumOfPDFeatures();
            int numOfAIFeatures = properties.getNumOfAIFeatures();
            int numOfACFeatures = properties.getNumOfACFeatures();
            int numOfGlobalFeatures= properties.getNumOfGlobalFeatures();
            int aiMaxBeamSize = properties.getNumOfAIBeamSize();
            int acMaxBeamSize = properties.getNumOfACBeamSize();
            String outputFile = properties.getOutputFilePathDev() + "_"+iter;
            String pdModelDir = properties.getPdModelDir();
            HashMap<String, Integer> globalReverseLabelMap = IO.load(properties.getGlobalReverseLabelMapPath());
            boolean usePI = properties.usePI();

            SupervisedSRL.Reranker.Decoder decoder = new SupervisedSRL.Reranker.Decoder(piClassifier, aiClassifier,
                    acClassifier, ap, indexMap, rerankerFeatureMap);
            decoder.decode(devSentences, numOfPIFeatures, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures,
                    aiMaxBeamSize, acMaxBeamSize, outputFile, aiCoefficient, pdModelDir, usePI);

            HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(globalReverseLabelMap);
            reverseLabelMap.put("0", reverseLabelMap.size());
            double f1= Evaluation.evaluate(outputFile, devSentences, indexMap, reverseLabelMap);

            if (f1 > bestFScore) {
                noImprovement = 0;
                bestFScore = f1;
                System.out.print("\nSaving the new model...");
                ap.saveModel(rerankerModelPath);
                System.out.println("Done!");
            } else {
                noImprovement++;
                if (noImprovement > 2) {
                    System.out.print("\nEarly stopping...");
                    break;
                }
            }
        }
    }

    private static int numOfFeatures(Properties properties) throws Exception {
        return IO.load(properties.getNumOfRerankerSeenFeaturesPath());
    }
}
