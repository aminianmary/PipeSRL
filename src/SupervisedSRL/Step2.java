package SupervisedSRL;

import SupervisedSRL.Reranker.RerankerInstanceGenerator;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.Properties;

import java.io.File;
import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step2 {
    public static void buildTrainDataPartitions(Properties properties) throws Exception {
        if (!properties.getSteps().contains(2) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 2 -- Creating Data Partitions\n>>>>>>>>>>>>>\n");
        String trainFilePath = properties.getTrainFile();
        int numOfPartitions = properties.getNumOfPartitions();
        RerankerInstanceGenerator rig = new RerankerInstanceGenerator(numOfPartitions);
        ArrayList<String>[] partitions = rig.getPartitions(trainFilePath);

        for (int partIdx = 0; partIdx < numOfPartitions; partIdx++) {
            String partitionDir = properties.getPartitionDir(partIdx);
            File file = new File(partitionDir);
            if (!file.exists()) {
                if (file.mkdir()) {
                    System.out.println(partitionDir + " is created!");
                } else {
                    System.out.println("Failed to create " + partitionDir);
                }
            }

            String trainDataPath = properties.getPartitionTrainDataPath(partIdx);
            String devDataPath = properties.getPartitionDevDataPath(partIdx);
            ArrayList<String> trainSentences = new ArrayList<>();

            for (int i = 0; i < numOfPartitions; i++)
                if (i != partIdx)
                    trainSentences.addAll(partitions[i]);
            ModelInfo.saveDataPartition(trainSentences, trainDataPath);
            ModelInfo.saveDataPartition(partitions[partIdx], devDataPath);
        }
    }
}
