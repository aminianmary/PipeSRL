package SupervisedSRL;

import SupervisedSRL.PI.PI;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Properties;
import util.IO;

import java.util.ArrayList;

public class Step7 {

    public static void predictPILabels (Properties properties) throws Exception{
        if (!properties.getSteps().contains(7))
            return;
        predictPILabels4EntireData(properties);
    }

    public static void predictPILabels4EntireData (Properties properties) throws Exception
    {
        if (!properties.getSteps().contains(7))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 7 -- Predicting Predicate Labels of Test data (used later as features)\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        String piModelPath = properties.getPiModelPath();
        String testFilePath = properties.getTestFile();
        String testPIAutoLabelsPath = properties.getTestPDLabelsPath();
        int numOfPIFeatures = properties.getNumOfPIFeatures();
        ArrayList<String> testSentences = IO.readCoNLLFile(testFilePath);
        IndexMap indexMap = IO.load(indexMapPath);
        System.out.print("\nMaking predictions on test data...\n");
        PI.predict(testSentences, indexMap, piModelPath, numOfPIFeatures, testPIAutoLabelsPath);
    }



}
