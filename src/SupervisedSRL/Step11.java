package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import ml.AveragedPerceptron;
import ml.RerankerAveragedPerceptron;
import util.IO;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Maryam Aminian on 9/12/16.
 */
public class Step11 {

    public static void decode(Properties properties)
            throws Exception {
        if (!properties.getSteps().contains(11))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 11 -- Decoding\n>>>>>>>>>>>>>\n");
        boolean usePI = properties.usePI();
        boolean supplement = properties.supplementOriginalLabels();
        AveragedPerceptron aiClassifier = AveragedPerceptron.loadModel(properties.getAiModelPath());
        AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(properties.getAcModelPath());
        AveragedPerceptron piClassifier = (usePI) ? AveragedPerceptron.loadModel(properties.getPiModelPath()) : null;
        IndexMap indexMap = IO.load(properties.getIndexMapFilePath());
        String pdModelDir = properties.getPdModelDir();
        ArrayList<String> testSentences = IO.readCoNLLFile(properties.getTestFile());
        int numOfPIFeatures = properties.getNumOfPIFeatures();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int numOfGlobalFeatures= properties.getNumOfGlobalFeatures();
        int aiMaxBeamSize = properties.getNumOfAIBeamSize();
        int acMaxBeamSize = properties.getNumOfACBeamSize();
        String testOutputFile = properties.getOutputFilePathTest();
        String testOutputFile_w_projected_info = properties.getOutputFilePathTest_w_projected_info();
        double aiCoefficient = properties.getAiCoefficient();

        if (properties.useReranker()) {
            HashMap<Object, Integer>[] rerankerFeatureMap = IO.load(properties.getRerankerFeatureMapPath());
            RerankerAveragedPerceptron reranker = RerankerAveragedPerceptron.loadModel(properties.getRerankerModelPath());
            SupervisedSRL.Reranker.Decoder decoder = new SupervisedSRL.Reranker.Decoder(piClassifier, aiClassifier, acClassifier,
                    reranker, indexMap, rerankerFeatureMap);
            System.out.println("\n>>>>>>>> Decoding Evaluation Data >>>>>>>>\n");
            decoder.decode(testSentences, numOfPIFeatures, numOfPDFeatures,numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures, aiMaxBeamSize, acMaxBeamSize,
                    testOutputFile, aiCoefficient, pdModelDir, usePI, supplement);
        } else {
            SupervisedSRL.Decoder decoder = new SupervisedSRL.Decoder(piClassifier, aiClassifier, acClassifier);
            System.out.println("\n>>>>>>>> Decoding Evaluation Data >>>>>>>>\n");
            decoder.decode(indexMap, testSentences, aiMaxBeamSize, acMaxBeamSize, numOfPIFeatures, numOfPDFeatures,
                    numOfAIFeatures,numOfACFeatures, testOutputFile,testOutputFile_w_projected_info,aiCoefficient, pdModelDir,usePI, supplement);
        }
    }
}
