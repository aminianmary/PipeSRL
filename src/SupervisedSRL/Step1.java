package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import util.IO;
import java.io.IOException;

/**
 * Created by Maryam Aminian on 9/9/16.
 * This step creates indexMap for the entire training data and saves it in indexMapFilePath (to be used in the next steps)
 */
public class Step1 {

    public static void Step1(String trainFilePath, String clusterFilePath, String indexMapFilePath) throws Exception{
            IndexMap indexMap= new IndexMap(IO.readCoNLLFile(trainFilePath), clusterFilePath);
            ModelInfo.saveIndexMap(indexMap, indexMapFilePath);
    }
}
