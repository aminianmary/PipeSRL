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
    public static void main(String[] args) {
        String trainFilePath = args[0];
        int numOfPartitions = Integer.parseInt(args[1]);
        String partitionsPath = args[2];

        RerankerInstanceGenerator rig = new RerankerInstanceGenerator(numOfPartitions);
        try {
            ArrayList<String>[] partitions= rig.getPartitions(trainFilePath);
            ModelInfo.saveDataPartitions(partitions, partitionsPath);
        }catch (IOException ioe){
            System.out.print(ioe.getMessage());
        }
    }
}
