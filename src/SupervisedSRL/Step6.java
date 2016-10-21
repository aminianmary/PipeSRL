package SupervisedSRL;

import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 10/3/16.
 */
public class Step6 {

    public static void predictPDLabels (Properties properties) throws Exception{
        if (!properties.getSteps().contains(6))
            return;
        predictPDLabels4EntireData(properties);
        if (properties.useReranker())
            predictPDLabels4Partitions(properties);
    }

    public static void predictPDLabels4EntireData (Properties properties) throws Exception
    {
        if (!properties.getSteps().contains(6))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 6.1 -- Predicting Predicate Labels of Train/dev data (used later as features)\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        String pdModelDir = properties.getPdModelDir();
        String trainFilePath = properties.getTrainFile();
        String devFilePath = properties.getDevFile();
        String testFilePath = properties.getTestFile();
        String trainPDAutoLabelsPath = properties.getTrainPDLabelsPath();
        String devPDAutoLabelsPath = properties.getDevPDLabelsPath();
        String testPDAutoLabelsPath = properties.getTestPDLabelsPath();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        ArrayList<String> trainSentences = IO.readCoNLLFile(trainFilePath);
        ArrayList<String> devSentences = IO.readCoNLLFile(devFilePath);
        ArrayList<String> testSentences = IO.readCoNLLFile(testFilePath);
        IndexMap indexMap = IO.load(indexMapPath);

        System.out.print("\nMaking predictions on train data...\n");
        PD.predict(trainSentences, indexMap, pdModelDir, numOfPDFeatures, trainPDAutoLabelsPath);
        System.out.print("\nMaking predictions on dev data...\n");
        PD.predict(devSentences, indexMap, pdModelDir, numOfPDFeatures, devPDAutoLabelsPath);
        System.out.print("\nMaking predictions on test data...\n");
        PD.predict(testSentences, indexMap, pdModelDir, numOfPDFeatures, testPDAutoLabelsPath);
    }

    public static void predictPDLabels4Partitions (Properties properties) throws Exception
    {
        if (!properties.getSteps().contains(6) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 6.2 -- Predicting Predicate Labels of Train/dev data partitions\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int numOfPartitions = properties.getNumOfPartitions();
        IndexMap indexMap = IO.load(indexMapPath);

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            System.out.println("\n>>>>>>>>\nPART "+devPartIdx+"\n>>>>>>>>\n");
            String pdModelDir = properties.getPartitionPdModelDir(devPartIdx);
            String trainFilePath = properties.getPartitionTrainDataPath(devPartIdx);
            String devFilePath = properties.getPartitionDevDataPath(devPartIdx);
            String devPDAutoLabelsPath = properties.getPartitionDevPDAutoLabelsPath(devPartIdx);
            String trainPDAutoLabelsPath = properties.getPartitionTrainPDAutoLabelsPath(devPartIdx);
            ArrayList<String> trainSentences = IO.load(trainFilePath);
            ArrayList<String> devSentences = IO.load(devFilePath);

            System.out.print("\nMaking predictions on train data...\n");
            PD.predict(trainSentences, indexMap, pdModelDir, numOfPDFeatures, trainPDAutoLabelsPath);
            System.out.print("\nMaking predictions on dev data...\n");
            PD.predict(devSentences, indexMap, pdModelDir, numOfPDFeatures, devPDAutoLabelsPath);
        }
    }
}
