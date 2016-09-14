package SupervisedSRL.Reranker;

import SupervisedSRL.Strcutures.Properties;
import ml.RerankerAveragedPerceptron;

import java.io.EOFException;
import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.HashSet;
import java.util.zip.GZIPInputStream;

/**
 * Created by Maryam Aminian on 8/25/16.
 */
public class Train {

    public static void trainReranker(Properties properties) throws Exception {
        int numOfPartitions = properties.getNumOfPartitions();
        int numOfTrainingIterations = properties.getMaxNumOfTrainingIterations();
        int numOfAIFeatures = properties.getNumOfAIFeatures();
        int numOfACFeatures = properties.getNumOfACFeatures();
        int numOfGlobalFeatures = properties.getNumOfGlobalFeatures();
        String rerankerModelPath = properties.getRerankerModelPath();
        int numOfRerankerFeatures = numOfAIFeatures + numOfACFeatures + numOfGlobalFeatures;

        HashSet<String> labels = new HashSet<String>();
        labels.add("1");
        RerankerAveragedPerceptron ap = new RerankerAveragedPerceptron(labels, numOfRerankerFeatures);

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
}
