package SupervisedSRL;

import SupervisedSRL.Reranker.Decoder;
import SupervisedSRL.Strcutures.IndexMap;
import ml.AveragedPerceptron;

import java.util.ArrayList;
import java.util.concurrent.ExecutorService;

/**
 * Created by monadiab on 9/12/16.
 */
public class Step7 {

    public static void Step7(AveragedPerceptron aiClassifier, AveragedPerceptron acClassifier, AveragedPerceptron reranker,
                             IndexMap indexMap, String devData, String pdModelDir, String outputFile,
                             int numOfPDFeatures, int numOfAIFeatures, int numOfACFeatures, int aiMaxBeamSize, int acMaxBeamSize)
    throws Exception{
        Decoder decoder = new Decoder(aiClassifier, acClassifier, reranker, indexMap, pdModelDir);
        decoder.decode(devData, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, aiMaxBeamSize, acMaxBeamSize, pdModelDir, outputFile);
    }
}
