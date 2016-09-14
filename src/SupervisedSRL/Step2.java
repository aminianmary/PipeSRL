package SupervisedSRL;

import SupervisedSRL.Reranker.RerankerInstanceGenerator;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.ProjectConstantPrefixes;
import SupervisedSRL.Strcutures.Properties;

import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step2 {
    public static void buildTrainDataPartitions(Properties properties) throws Exception {
        String trainFilePath = properties.getTrainFile();
        int numOfPartitions= properties.getNumOfPartitions();
        RerankerInstanceGenerator rig = new RerankerInstanceGenerator(numOfPartitions);
        ArrayList<String>[] partitions = rig.getPartitions(trainFilePath);

        for (int partIdx = 0; partIdx < numOfPartitions; partIdx++) {
            String trainDataPath = properties.getPartitionTrainDataPath(partIdx);
            String devDataPath = properties.getPartitionDevDataPath(partIdx);
            ArrayList<String> trainSentences = new ArrayList<>();

            for (int i=0; i< numOfPartitions ; i++)
                if (i != partIdx)
                    trainSentences.addAll(partitions[i]);
            ModelInfo.saveDataPartition(trainSentences, trainDataPath);
            ModelInfo.saveDataPartition(partitions[partIdx], devDataPath);
        }
    }
}
