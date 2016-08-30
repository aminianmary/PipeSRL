package SupervisedSRL.Reranker;

import SupervisedSRL.Strcutures.CompactArray;
import ml.AveragedPerceptron;

import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.zip.GZIPInputStream;

/**
 * Created by Maryam Aminian on 8/25/16.
 */
public class Train {

    public static String trainReranker(int numOfParts, String rerankerInstanceFilePrefix, int numOfTrainingIterations,
                                int numOfRerankerFeatures, String modelDir) throws Exception{
        HashSet<String> labels= new HashSet<String>();
        labels.add("1");
        AveragedPerceptron ap = new AveragedPerceptron(labels, numOfRerankerFeatures);

        for (int iter =0; iter< numOfTrainingIterations; iter++){
            for (int devPart =0; devPart< numOfParts; devPart++){
                FileInputStream fis = new FileInputStream(rerankerInstanceFilePrefix+devPart);
                GZIPInputStream gz = new GZIPInputStream(fis);
                ObjectInput reader = new ObjectInputStream(gz);
                ArrayList<RerankerPool> rerankerPools = (ArrayList<RerankerPool>) reader.readObject();

                for (RerankerPool pool: rerankerPools)
                    ap.learnInstance(pool);
            }
        }
        String rerankerModelPath = modelDir+"/reranker.model";
        ap.saveModel(rerankerModelPath);
        return rerankerModelPath;
    }
}
