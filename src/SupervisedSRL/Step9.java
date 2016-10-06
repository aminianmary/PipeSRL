package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import SupervisedSRL.Strcutures.RerankerFeatureMap;
import ml.AveragedPerceptron;
import ml.RerankerAveragedPerceptron;
import util.IO;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by monadiab on 9/12/16.
 */
public class Step9 {

    public static void decode(Properties properties)
            throws Exception {
        if (!properties.getSteps().contains(9))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 9 -- Decoding\n>>>>>>>>>>>>>\n");
        AveragedPerceptron aiClassifier = AveragedPerceptron.loadModel(properties.getAiModelPath());
        AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(properties.getAcModelPath());
        IndexMap indexMap = IO.load(properties.getIndexMapFilePath());
        String pdAutoLabelsPathDev = properties.getDevAutoPDLabelsPath();
        String pdAutoLabelsPathTest = properties.getTestAutoPDLabelsPath();
        ArrayList<String> devSentences = IO.readCoNLLFile(properties.getDevFile());
        ArrayList<String> testSentences = IO.readCoNLLFile(properties.getTestFile());
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int numOfGlobalFeatures= properties.getNumOfGlobalFeatures();
        int aiMaxBeamSize = properties.getNumOfAIBeamSize();
        int acMaxBeamSize = properties.getNumOfACBeamSize();
        String devOutputFile = properties.getOutputFilePathDev();
        String testOutputFile = properties.getOutputFilePathTest();
        double aiCoefficient = properties.getAiCoefficient();
        if (properties.useReranker()) {
            HashMap<Object, Integer>[] rerankerFeatureMap = IO.load(properties.getRerankerFeatureMapPath());
            RerankerAveragedPerceptron reranker = RerankerAveragedPerceptron.loadModel(properties.getRerankerModelPath());
            SupervisedSRL.Reranker.Decoder decoder = new SupervisedSRL.Reranker.Decoder(aiClassifier, acClassifier,
                    reranker, indexMap, rerankerFeatureMap);
            System.out.println("\n>>>>>>>> Decoding Development Data >>>>>>>>\n");
            decoder.decode(devSentences,numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures, aiMaxBeamSize, acMaxBeamSize,
                    devOutputFile, aiCoefficient, pdAutoLabelsPathDev);

            System.out.println("\n>>>>>>>> Decoding Evaluation Data >>>>>>>>\n");
            decoder.decode(testSentences,numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures, aiMaxBeamSize, acMaxBeamSize,
                    testOutputFile, aiCoefficient, pdAutoLabelsPathTest);
        } else {
            SupervisedSRL.Decoder decoder = new SupervisedSRL.Decoder(aiClassifier, acClassifier);
            System.out.println("\n>>>>>>>> Decoding Development Data >>>>>>>>\n");
            decoder.decode(indexMap, devSentences, aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures,numOfACFeatures,
                    devOutputFile,aiCoefficient, pdAutoLabelsPathDev);

            System.out.println("\n>>>>>>>> Decoding Evaluation Data >>>>>>>>\n");
            decoder.decode(indexMap, testSentences, aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures,numOfACFeatures,
                    testOutputFile,aiCoefficient, pdAutoLabelsPathTest);
        }
    }
}
