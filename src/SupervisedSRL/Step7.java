package SupervisedSRL;

import SupervisedSRL.Reranker.Train;
import SupervisedSRL.Strcutures.Properties;

/**
 * Created by Maryam Aminian on 9/12/16.
 */
public class Step7 {
    public static void buildRerankerModel(Properties properties)
            throws Exception {
        if (!properties.getSteps().contains(7) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 7 -- Build Reranker Model\n>>>>>>>>>>>>>\n");
        Train.trainReranker(properties);
    }
}
