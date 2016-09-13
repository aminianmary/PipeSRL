package SupervisedSRL.Reranker;

import ml.RerankerAveragedPerceptron;

import java.io.EOFException;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.HashSet;

/**
 * Created by Maryam Aminian on 8/25/16.
 */
public class Train {

    public static void trainReranker(int numOfParts, String rerankerInstanceFilePrefix, int numOfTrainingIterations,
                                     int numOfRerankerFeatures, String rerankerModelPath) throws Exception {
        HashSet<String> labels = new HashSet<String>();
        labels.add("1");
        RerankerAveragedPerceptron ap = new RerankerAveragedPerceptron(labels, numOfRerankerFeatures);

        for (int iter = 0; iter < numOfTrainingIterations; iter++) {
            System.out.println("Iteration " + iter + "\n>>>>>>>>>>>\n");
            for (int devPart = 0; devPart < numOfParts; devPart++) {
                System.out.println("Loading/learning train instances for dev part " + devPart + "\n");
                FileInputStream fis = new FileInputStream(rerankerInstanceFilePrefix + devPart);
                ObjectInputStream reader = new ObjectInputStream(fis);

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
