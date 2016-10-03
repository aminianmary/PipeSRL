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
        String pdAutoLabelsPath = properties.getDevAutoPDLabelsPath();
        ArrayList<String> devSentences = IO.readCoNLLFile(properties.getDevFile());
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int numOfGlobalFeatures= properties.getNumOfGlobalFeatures();
        int aiMaxBeamSize = properties.getNumOfAIBeamSize();
        int acMaxBeamSize = properties.getNumOfACBeamSize();
        String outputFile = properties.getOutputFilePath();
        double aiCoefficient = properties.getAiCoefficient();
        if (properties.useReranker()) {
            HashMap<Object, Integer>[] rerankerFeatureMap = IO.load(properties.getRerankerFeatureMapPath());
            RerankerAveragedPerceptron reranker = RerankerAveragedPerceptron.loadModel(properties.getRerankerModelPath());
            SupervisedSRL.Reranker.Decoder decoder = new SupervisedSRL.Reranker.Decoder(aiClassifier, acClassifier, reranker, indexMap, rerankerFeatureMap);
            decoder.decode(devSentences,numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures, aiMaxBeamSize, acMaxBeamSize, outputFile, aiCoefficient, pdAutoLabelsPath);
        } else {
            SupervisedSRL.Decoder decoder = new SupervisedSRL.Decoder(aiClassifier, acClassifier);
            decoder.decode(indexMap, devSentences, aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures,numOfACFeatures,outputFile,aiCoefficient, pdAutoLabelsPath);
        }
    }
}
