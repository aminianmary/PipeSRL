package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import util.IO;
import java.io.IOException;

/**
 * Created by Maryam Aminian on 9/9/16.
 */
public class Step1 {
    public static void main(String[] args) {
        String trainFilePath = args[0]; //input 1
        String clusterFilePath = args[1]; //input 2
        String indexMapFilePath = args[2]; //input 3
        try {
            IndexMap indexMap= new IndexMap(IO.readCoNLLFile(trainFilePath), clusterFilePath);
            ModelInfo.saveIndexMap(indexMap, indexMapFilePath);
        }catch (IOException ioe){
            System.out.print(ioe.getMessage());
        }
    }
}
