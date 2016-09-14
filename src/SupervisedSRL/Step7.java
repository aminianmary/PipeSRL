package SupervisedSRL;

import SupervisedSRL.Reranker.Decoder;
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
        AveragedPerceptron aiClassifier = AveragedPerceptron.loadModel(properties.getAiModelPath());
        AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(properties.getAcModelPath());
        IndexMap indexMap = ModelInfo.loadIndexMap(properties.getIndexMapFilePath());
        HashMap<Object, Integer>[] rerankerFeatureMap = ModelInfo.loadFeatureMap(properties.getRerankerFeatureMapPath());
        String pdModelDir = properties.getPdModelDir();
        ArrayList<String> devSentences = IO.readCoNLLFile(properties.getDevFile());
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int aiMaxBeamSize = properties.getNumOfAIBeamSize();
        int acMaxBeamSize = properties.getNumOfACBeamSize();
        String outputFile = properties.getOutputFilePath();

        RerankerAveragedPerceptron reranker = RerankerAveragedPerceptron.loadModel(properties.getRerankerModelPath());
        Decoder decoder = new Decoder(aiClassifier, acClassifier, reranker, indexMap, rerankerFeatureMap, pdModelDir);
        decoder.decode(devSentences, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, aiMaxBeamSize, acMaxBeamSize, pdModelDir, outputFile);
    }
}
