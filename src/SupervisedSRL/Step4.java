package SupervisedSRL;

import SupervisedSRL.PI.PI;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 10/21/16.
 */
public class Step4 {
    public static void predictPILabels(Properties properties) throws Exception {
        if (!properties.getSteps().contains(4))
            return;
        predictPILabels4EntireData(properties);
        if (properties.useReranker())
            predictPILabels4Partitions(properties);
    }

    public static void predictPILabels4EntireData(Properties properties) throws Exception {
        if (!properties.getSteps().contains(4))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 4.1 -- Predicting Predicate of Train/dev data\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        String piModelPath = properties.getPiModelPath();
        String trainFilePath = properties.getTrainFile();
        String devFilePath = properties.getDevFile();
        String testFilePath = properties.getTestFile();
        String trainPILabelsPath = properties.getTrainPILabelsPath();
        String devPILabelsPath = properties.getDevPILabelsPath();
        String testPILabelsPath = properties.getTestPILabelsPath();
        int numOfPIFeatures = properties.getNumOfPIFeatures();
        ArrayList<String> trainSentences = IO.readCoNLLFile(trainFilePath);
        ArrayList<String> devSentences = IO.readCoNLLFile(devFilePath);
        ArrayList<String> testSentences = IO.readCoNLLFile(testFilePath);
        IndexMap indexMap = IO.load(indexMapPath);

        System.out.print("\nMaking predictions on train data...\n");
        PI.predict(trainSentences, indexMap, piModelPath, numOfPIFeatures, trainPILabelsPath);
        System.out.print("\nMaking predictions on dev data...\n");
        PI.predict(devSentences, indexMap, piModelPath, numOfPIFeatures, devPILabelsPath);
        System.out.print("\nMaking predictions on test data...\n");
        PI.predict(testSentences, indexMap, piModelPath, numOfPIFeatures, testPILabelsPath);
    }

    public static void predictPILabels4Partitions(Properties properties) throws Exception {
        if (!properties.getSteps().contains(4) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 4.2 -- Predicting Predicate of Train/dev data partitions\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        int numOfPIFeatures = properties.getNumOfPIFeatures();
        int numOfPartitions = properties.getNumOfPartitions();
        IndexMap indexMap = IO.load(indexMapPath);

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            System.out.println("\n>>>>>>>>\nPART " + devPartIdx + "\n>>>>>>>>\n");
            String piModelPath = properties.getPartitionPiModelPath(devPartIdx);
            String trainFilePath = properties.getPartitionTrainDataPath(devPartIdx);
            String devFilePath = properties.getPartitionDevDataPath(devPartIdx);
            String devPILabels = properties.getPartitionDevPILabelsPath(devPartIdx);
            String trainPILabels = properties.getPartitionTrainPILabelsPath(devPartIdx);
            ArrayList<String> trainSentences = IO.load(trainFilePath);
            ArrayList<String> devSentences = IO.load(devFilePath);

            System.out.print("\nMaking predictions on train data...\n");
            PI.predict(trainSentences, indexMap, piModelPath, numOfPIFeatures, trainPILabels);
            System.out.print("\nMaking predictions on dev data...\n");
            PI.predict(devSentences, indexMap, piModelPath, numOfPIFeatures, devPILabels);
        }
    }
}
