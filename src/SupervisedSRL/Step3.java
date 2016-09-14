package SupervisedSRL;

import SupervisedSRL.Strcutures.*;
import ml.AveragedPerceptron;
import util.IO;

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
    public static void buildModel4EntireData(Properties properties) throws Exception {
        String indexMapPath = properties.getIndexMapFilePath();
        String pdModelDir =properties.getPdModelDir() ;
        String aiModelPath =properties.getAiModelPath() ;
        String acModelPath =properties.getAcModelPath();
        String trainFilePath = properties.getTrainFile();
        String devFilePath =  properties.getDevFile();
        int maxTrainingIters = properties.getMaxNumOfTrainingIterations();
        int numOfAIFeatures= properties.getNumOfAIFeatures();
        int numOfACFeatures= properties.getNumOfACFeatures();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int aiBeamSize= properties.getNumOfAIBeamSize();
        int acBeamSize= properties.getNumOfACBeamSize();

        ArrayList<String> trainSentences = IO.readCoNLLFile(trainFilePath);
        ArrayList<String> devSentences = IO.readCoNLLFile(devFilePath);

        IndexMap indexMap = ModelInfo.loadIndexMap(indexMapPath);
        boolean isModelBuiltOnEntireTrainData = true;
        Train.train(trainSentences, devSentences, pdModelDir, aiModelPath, acModelPath, indexMap, maxTrainingIters,
                numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiBeamSize, acBeamSize, isModelBuiltOnEntireTrainData);
    }

    public static void buildModel4Partitions(Properties properties) throws Exception {
        String indexMapPath = properties.getIndexMapFilePath();
        int maxTrainingIters = properties.getMaxNumOfTrainingIterations();
        int numOfAIFeatures= properties.getNumOfAIFeatures();
        int numOfACFeatures= properties.getNumOfACFeatures();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int aiBeamSize= properties.getNumOfAIBeamSize();
        int acBeamSize= properties.getNumOfACBeamSize();
        int numOfPartitions = properties.getNumOfPartitions();
        IndexMap indexMap = ModelInfo.loadIndexMap(indexMapPath);

        for (int devPartIdx =0 ; devPartIdx < numOfPartitions ; devPartIdx++) {
            String pdModelDir = properties.getPartitionPdModelDir(devPartIdx);
            String aiModelPath = properties.getPartitionAIModelPath(devPartIdx);
            String acModelPath = properties.getPartitionACModelPath(devPartIdx);
            String trainFilePath =  properties.getPartitionTrainDataPath(devPartIdx);
            String devFilePath = properties.getPartitionDevDataPath(devPartIdx);
            ArrayList<String> trainSentences = IO.readCoNLLFile(trainFilePath);
            ArrayList<String> devSentences = IO.readCoNLLFile(devFilePath);
            boolean isModelBuiltOnEntireTrainData = false;
            Train.train(trainSentences, devSentences, pdModelDir, aiModelPath, acModelPath, indexMap, maxTrainingIters,
                    numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiBeamSize, acBeamSize, isModelBuiltOnEntireTrainData);
        }
    }

}
