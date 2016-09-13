package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.Pair;
import ml.AveragedPerceptron;

import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step3 {
    public static void buildModels(ArrayList<String> trainSentences, ArrayList<String> devSentences, String indexMapPath, String pdModelDir,
                                   String aiModelPath, String acModelPath, int maxTrainingIters, int numOfPDFeatures,
                                   int numOfAIFeatures, int numOfACFeatures, int aiBeamSize, int acBeamSize, boolean saveReverseLabelMap) throws Exception {
        IndexMap indexMap = ModelInfo.loadIndexMap(indexMapPath);
        Train.train(trainSentences, devSentences, pdModelDir, aiModelPath, acModelPath, indexMap, maxTrainingIters,
                numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiBeamSize, acBeamSize, saveReverseLabelMap);
    }

    public static Pair<AveragedPerceptron, AveragedPerceptron> loadModels(String aiModelPath, String acModelPath) throws Exception {
        AveragedPerceptron aiClassifier = AveragedPerceptron.loadModel(aiModelPath);
        AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(acModelPath);
        return new Pair<>(aiClassifier, acClassifier);
    }

    public static HashMap<String, Integer> loadReverseLabelMap(String reverseLabelMapPath) throws Exception {
        FileInputStream fis = new FileInputStream(reverseLabelMapPath);
        GZIPInputStream gz = new GZIPInputStream(fis);
        ObjectInput reader = new ObjectInputStream(gz);
        return (HashMap<String, Integer>) reader.readObject();
    }
}
