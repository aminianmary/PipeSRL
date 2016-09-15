package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.Properties;
import ml.AveragedPerceptron;
import ml.RerankerAveragedPerceptron;
import util.IO;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by monadiab on 9/12/16.
 */
public class Step7 {

    public static void decode(Properties properties)
            throws Exception {
        if (!properties.getSteps().contains(7))
            return;
        AveragedPerceptron aiClassifier = AveragedPerceptron.loadModel(properties.getAiModelPath());
        AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(properties.getAcModelPath());
        IndexMap indexMap = ModelInfo.load(properties.getIndexMapFilePath());
        String pdModelDir = properties.getPdModelDir();
        ArrayList<String> devSentences = IO.readCoNLLFile(properties.getDevFile());
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int aiMaxBeamSize = properties.getNumOfAIBeamSize();
        int acMaxBeamSize = properties.getNumOfACBeamSize();
        String outputFile = properties.getOutputFilePath();

        if (properties.useReranker()) {
            HashMap<Object, Integer>[] rerankerFeatureMap = ModelInfo.load(properties.getRerankerFeatureMapPath());
            RerankerAveragedPerceptron reranker = RerankerAveragedPerceptron.loadModel(properties.getRerankerModelPath());
            SupervisedSRL.Reranker.Decoder decoder = new SupervisedSRL.Reranker.Decoder(aiClassifier, acClassifier, reranker, indexMap, rerankerFeatureMap, pdModelDir);
            decoder.decode(devSentences, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, aiMaxBeamSize, acMaxBeamSize, pdModelDir, outputFile);
        } else {
            SupervisedSRL.Decoder decoder = new SupervisedSRL.Decoder(aiClassifier, acClassifier);
            decoder.decode(indexMap, devSentences, aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures, pdModelDir, outputFile);
        }
    }
}
