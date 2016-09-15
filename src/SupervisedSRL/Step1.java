package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

/**
 * Created by Maryam Aminian on 9/9/16.
 * This step creates indexMap for the entire training data and saves it in indexMapFilePath (to be used in the next steps)
 */
public class Step1 {

    public static void buildIndexMap(Properties properties) throws Exception {
        if (!properties.getSteps().contains(1))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 1 -- building IndexMap\n>>>>>>>>>>>>>\n");
        String trainFilePath = properties.getTrainFile();
        String clusterFilePath = properties.getClusterFile();
        String indexMapFilePath = properties.getIndexMapFilePath();
        IndexMap indexMap = new IndexMap(trainFilePath, clusterFilePath);
        IO.write(indexMap, indexMapFilePath);
    }
}
