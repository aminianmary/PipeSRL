package SupervisedSRL;

import SupervisedSRL.Reranker.Train;
import SupervisedSRL.Strcutures.Properties;

/**
 * Created by Maryam Aminian on 9/12/16.
 */
public class Step6 {
    public static void buildRerankerModel(Properties properties)
            throws Exception {
        if (!properties.getSteps().contains(6) || !properties.useReranker())
            return;
        System.out.println("Step 6 -- Build Reranker Model");
        Train.trainReranker(properties);
    }
}
