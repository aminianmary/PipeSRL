package SupervisedSRL;

import SupervisedSRL.Reranker.RerankerInstanceGenerator;
import SupervisedSRL.Strcutures.ModelInfo;

import java.io.IOException;
import java.net.Inet4Address;
import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 9/9/16.
 *
 */
public class Step2 {
    public static void Step2(String trainFilePath, String partitionsPath, int numOfPartitions) throws Exception{
        RerankerInstanceGenerator rig = new RerankerInstanceGenerator(numOfPartitions);
        ArrayList<String>[] partitions= rig.getPartitions(trainFilePath);
        ModelInfo.saveDataPartitions(partitions, partitionsPath);
    }
}
