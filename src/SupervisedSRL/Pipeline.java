package SupervisedSRL;

import SentenceStruct.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.Pair;
import SupervisedSRL.Strcutures.ProjectConstantPrefixes;
import ml.AveragedPerceptron;
import util.IO;
import util.StringUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;

/**
 * Created by Maryam Aminian on 9/12/16.
 */
public class Pipeline {

    public final static int numOfPDFeatures = 9;
    public final static int numOfAIFeatures = 25;
    public final static int numOfACFeatures = 25;
    public final static int numOfGlobalFeatures = 1;

    public static void main(String[] args) {
        String trainFile = args[0];
        String devFile = args[1];
        String clusterFile = args[2];
        String modelDir = args[3];
        int numOfPartitions = Integer.parseInt(args[4]);
        int maxNumOfTrainingIterations = Integer.parseInt(args[5]);
        int numOfAIBeamSize = Integer.parseInt(args[6]);
        int numOfACBeamSize = Integer.parseInt(args[7]);

         Properties properties = new Properties(trainFile, devFile, clusterFile,modelDir, numOfPartitions,
                 maxNumOfTrainingIterations,numOfAIBeamSize, numOfACBeamSize,
                 numOfPDFeatures, numOfAIFeatures, numOfACFeatures,numOfGlobalFeatures);
        try {
            Step1.buildIndexMap(properties);
            Step2.buildTrainDataPartitions(properties);
            Step3.buildModel4EntireData(properties);
            Step3.buildModel4Partitions(properties);
            Step4.buildRerankerFeatureMap(properties);
            Step5.generateRerankerInstances(properties);
            Step6.buildRerankerModel(properties);
            Step7.decode(properties);
            Step8.evaluate(properties);
        } catch (Exception e) {
            System.out.print(e.getMessage());
        }
    }

}
