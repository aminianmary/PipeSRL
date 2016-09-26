package SupervisedSRL;

import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

import java.util.ArrayList;
import java.util.concurrent.ExecutorService;

/**
 * Created by Maryam Aminian on 9/26/16.
 */
public class Step3 {
    public static void buildPDModel4EntireData (Properties properties) throws Exception{
        if (!properties.getSteps().contains(3))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 3.1 -- Building PD models on entire data\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        String pdModelDir = properties.getPdModelDir();
        String trainFilePath = properties.getTrainFile();
        String devFilePath = properties.getDevFile();
        int maxTrainingIters = properties.getMaxNumOfPDTrainingIterations();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        ArrayList<String> trainSentences = IO.readCoNLLFile(trainFilePath);
        ArrayList<String> devSentences = IO.readCoNLLFile(devFilePath);
        IndexMap indexMap = IO.load(indexMapPath);

        PD.train(trainSentences, devSentences, indexMap, maxTrainingIters, pdModelDir, numOfPDFeatures);
    }

    public static void buildPDModel4Partitions(Properties properties) throws Exception {
        if (!properties.getSteps().contains(3) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 3.2 -- Building PD models on partitions\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        int maxTrainingIters = properties.getMaxNumOfPDTrainingIterations();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int numOfPartitions = properties.getNumOfPartitions();
        IndexMap indexMap = IO.load(indexMapPath);

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            System.out.println("\n>>>>>>>>\nPART "+devPartIdx+"\n>>>>>>>>\n");
            String pdModelDir = properties.getPartitionPdModelDir(devPartIdx);
            String trainFilePath = properties.getPartitionTrainDataPath(devPartIdx);
            String devFilePath = properties.getPartitionDevDataPath(devPartIdx);
            ArrayList<String> trainSentences = IO.load(trainFilePath);
            ArrayList<String> devSentences = IO.load(devFilePath);

            PD.train(trainSentences, devSentences, indexMap, maxTrainingIters, pdModelDir, numOfPDFeatures);
        }

    }
}
