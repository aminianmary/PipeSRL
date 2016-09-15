package SupervisedSRL.Reranker;

import SupervisedSRL.Strcutures.Properties;
import SupervisedSRL.Strcutures.RerankerFeatureMap;
import ml.RerankerAveragedPerceptron;
import util.IO;

import java.io.EOFException;
import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.zip.GZIPInputStream;

/**
 * Created by Maryam Aminian on 8/25/16.
 */
public class Train {
    public static void trainReranker(Properties properties) throws Exception {
        int numOfPartitions = properties.getNumOfPartitions();
        int numOfTrainingIterations = properties.getMaxNumOfTrainingIterations();
        String rerankerModelPath = properties.getRerankerModelPath();

        RerankerAveragedPerceptron ap = new RerankerAveragedPerceptron(numOfFeatures(properties));

        for (int iter = 0; iter < numOfTrainingIterations; iter++) {
            System.out.println("Iteration " + iter + "\n>>>>>>>>>>>\n");
            for (int devPart = 0; devPart < numOfPartitions; devPart++) {
                System.out.println("Loading/learning train instances for dev part " + devPart + "\n");
                FileInputStream fis = new FileInputStream(properties.getRerankerInstancesFilePath(devPart));
                GZIPInputStream gz = new GZIPInputStream(fis);
                ObjectInput reader = new ObjectInputStream(gz);

                while (true) {
                    try {
                        RerankerPool pool = (RerankerPool) reader.readObject();
                        ap.learnInstance(pool);
                    } catch (EOFException e) {
                        reader.close();
                        break;
                    }
                }
                System.out.println("Part " + devPart + " done!");
            }
        }
        ap.saveModel(rerankerModelPath);
    }

    // todo this is not efficient.
    private static int numOfFeatures(Properties properties) throws Exception {
        return ((RerankerFeatureMap) IO.load(properties.getRerankerFeatureMapPath())).getNumOfSeenFeatures();
    }
}
