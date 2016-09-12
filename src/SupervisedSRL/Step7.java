package SupervisedSRL;

import SupervisedSRL.Reranker.Decoder;
import SupervisedSRL.Strcutures.IndexMap;
import ml.AveragedPerceptron;
import ml.RerankerAveragedPerceptron;

import java.util.HashMap;

/**
 * Created by monadiab on 9/12/16.
 */
public class Step7 {

    public static void Step7(AveragedPerceptron aiClassifier, AveragedPerceptron acClassifier, RerankerAveragedPerceptron reranker,
                             IndexMap indexMap, HashMap<Object, Integer>[] rerankerFeatureMap, String devData, String pdModelDir, String outputFile,
                             int numOfPDFeatures, int numOfAIFeatures, int numOfACFeatures, int aiMaxBeamSize, int acMaxBeamSize)
    throws Exception{
        Decoder decoder = new Decoder(aiClassifier, acClassifier, reranker, indexMap, rerankerFeatureMap, pdModelDir);
        decoder.decode(devData, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, aiMaxBeamSize, acMaxBeamSize, pdModelDir, outputFile);
    }
}
