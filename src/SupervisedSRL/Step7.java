package SupervisedSRL;

import SupervisedSRL.Reranker.Decoder;
import SupervisedSRL.Strcutures.IndexMap;
import ml.AveragedPerceptron;
import ml.RerankerAveragedPerceptron;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by monadiab on 9/12/16.
 */
public class Step7 {

    public static void decode(String aiModelPath, String acModelPath, String rerankerModelPath,
                              IndexMap indexMap, HashMap<Object, Integer>[] rerankerFeatureMap, ArrayList<String> devSentences, String pdModelDir, String outputFile,
                              int numOfPDFeatures, int numOfAIFeatures, int numOfACFeatures, int aiMaxBeamSize, int acMaxBeamSize)
            throws Exception {
        AveragedPerceptron aiClassifier = AveragedPerceptron.loadModel(aiModelPath);
        AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(acModelPath);
        RerankerAveragedPerceptron reranker = RerankerAveragedPerceptron.loadModel(rerankerModelPath);

        Decoder decoder = new Decoder(aiClassifier, acClassifier, reranker, indexMap, rerankerFeatureMap, pdModelDir);
        decoder.decode(devSentences, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, aiMaxBeamSize, acMaxBeamSize, pdModelDir, outputFile);
    }
}
