package SupervisedSRL;

import SentenceStruct.Sentence;
import SentenceStruct.simplePA;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Pair;
import SupervisedSRL.Strcutures.Prediction4Reranker;
import SupervisedSRL.Strcutures.Properties;
import ml.AveragedPerceptron;
import util.IO;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Maryam Aminian on 10/3/16.
 */
public class Step7 {

    public static void predictPDLabels (Properties properties) throws Exception{
        if (!properties.getSteps().contains(7))
            return;
        predictPDLabels4EntireData(properties);
        if (properties.useReranker())
            predictPDLabels4Partitions(properties);
    }

    public static void predictPDLabels4EntireData (Properties properties) throws Exception
    {
        if (!properties.getSteps().contains(7))
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 7.1 -- Predicting Predicate Labels of Train/dev data (used later as features)\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        String pdModelDir = properties.getPdModelDir();
        String testFilePath = properties.getTestFile();
        String testPDAutoLabelsPath = properties.getTestPDLabelsPath();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int numOfPIFeatures = properties.getNumOfPIFeatures();
        ArrayList<String> testSentences = IO.readCoNLLFile(testFilePath);
        IndexMap indexMap = IO.load(indexMapPath);
        boolean usePI = properties.usePI();

        System.out.print("\nMaking predictions on test data...\n");
        if (!usePI)
            PD.predict(testSentences, indexMap, pdModelDir, numOfPDFeatures, testPDAutoLabelsPath);
        else {
            AveragedPerceptron piClassifier = AveragedPerceptron.loadModel(properties.getPiModelPath());
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(testPDAutoLabelsPath+".plain")));

            for (String sentence: testSentences){
                Sentence s =new Sentence(sentence, indexMap);
                int[] sentenceLemmas = s.getLemmas();
                String[] sentenceLemmas_str = s.getLemmas_str();

                for (int wordIdx = 1; wordIdx < s.getLength(); wordIdx++) {
                    boolean isPredicate = false;
                    Object[] featureVector = FeatureExtractor.extractPIFeatures(wordIdx, s, numOfPIFeatures, indexMap);
                    String piPrediction = piClassifier.predict(featureVector);
                    if (piPrediction.equals("1"))
                        isPredicate = true;
                    if (isPredicate) {
                        //identified as a predicate
                        int pIdx = wordIdx;
                        int plem = sentenceLemmas[pIdx];
                        String pLabel = "";

                        Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, s, numOfPDFeatures, indexMap);
                        File f1 = new File(pdModelDir + "/" + plem);
                        if (f1.exists() && !f1.isDirectory()) {
                            //seen predicates
                            AveragedPerceptron classifier = AveragedPerceptron.loadModel(pdModelDir + "/" + plem);
                            pLabel = classifier.predict(pdfeats);
                        } else {
                            if (plem != indexMap.unknownIdx) {
                                pLabel = indexMap.int2str(plem) + ".01"; //seen pLem
                            } else {
                                pLabel = sentenceLemmas_str[pIdx] + ".01"; //unseen pLem
                            }
                        }
                        writer.write(wordIdx+"\tY\t"+pLabel+"\n");
                    }
                    else
                        writer.write(wordIdx+"\t_\t_\n");
                }
                writer.write("\n");
            }
            writer.flush();
            writer.close();
        }
    }

    public static void predictPDLabels4Partitions (Properties properties) throws Exception
    {
        if (!properties.getSteps().contains(6) || !properties.useReranker())
            return;
        System.out.println("\n>>>>>>>>>>>>>\nStep 7.2 -- Predicting Predicate Labels of Train/dev data partitions\n>>>>>>>>>>>>>\n");
        String indexMapPath = properties.getIndexMapFilePath();
        int numOfPDFeatures = properties.getNumOfPDFeatures();
        int numOfPartitions = properties.getNumOfPartitions();
        IndexMap indexMap = IO.load(indexMapPath);

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            System.out.println("\n>>>>>>>>\nPART "+devPartIdx+"\n>>>>>>>>\n");
            String pdModelDir = properties.getPartitionPdModelDir(devPartIdx);
            String trainFilePath = properties.getPartitionTrainDataPath(devPartIdx);
            String devFilePath = properties.getPartitionDevDataPath(devPartIdx);
            String devPDAutoLabelsPath = properties.getPartitionDevPDAutoLabelsPath(devPartIdx);
            String trainPDAutoLabelsPath = properties.getPartitionTrainPDAutoLabelsPath(devPartIdx);
            ArrayList<String> trainSentences = IO.load(trainFilePath);
            ArrayList<String> devSentences = IO.load(devFilePath);

            System.out.print("\nMaking predictions on train data...\n");
            PD.predict(trainSentences, indexMap, pdModelDir, numOfPDFeatures, trainPDAutoLabelsPath);
            if (devSentences.size()>0) {
                System.out.print("\nMaking predictions on dev data...\n");
                PD.predict(devSentences, indexMap, pdModelDir, numOfPDFeatures, devPDAutoLabelsPath);
            }
        }
    }
}
