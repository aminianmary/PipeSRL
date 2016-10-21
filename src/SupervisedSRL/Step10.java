package SupervisedSRL;

import SupervisedSRL.Reranker.Train;
import SupervisedSRL.Strcutures.Properties;

/**
 * Created by Maryam Aminian on 9/12/16.
 */
public class Step10 {
    public static void trainRerankerModel(Properties properties)
            throws Exception {
        if (!properties.getSteps().contains(10) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 10 -- Build Reranker Model\n>>>>>>>>>>>>>\n");
        Train.trainReranker(properties);
    }
}
