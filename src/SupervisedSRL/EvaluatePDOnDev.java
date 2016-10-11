package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import util.IO;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Maryam Aminian on 10/7/16.
 */
public class EvaluatePDOnDev {
    public static void main(String[] args) throws Exception {
        String trainData = args[0];
        String devData = args[1];
        String evalData = args[2];
        String evalDataPredicatePredictionsFilePath = args[3];
        String indexMapFilePath = args[4];
        int numOfPDFeaturs = Integer.parseInt(args[5]);
        boolean readPredictionsFromOutputFile = Boolean.parseBoolean(args[6]);


        IndexMap indexMap = IO.load(indexMapFilePath);
        ArrayList<String> trainSentences = IO.readCoNLLFile(trainData);
        ArrayList<String> devSentences = IO.readCoNLLFile(devData);
        ArrayList<String> goldEvalSentences = IO.readCoNLLFile(evalData);
        HashMap<Integer, String>[] evalDataPredicatePredictions = (!readPredictionsFromOutputFile) ?
                (HashMap<Integer, String>[]) IO.load(evalDataPredicatePredictionsFilePath) :
                IO.getDisambiguatedPredicatesFromOutput(evalDataPredicatePredictionsFilePath, goldEvalSentences.size());;

        Evaluation.evaluatePD(trainSentences, devSentences, goldEvalSentences, evalDataPredicatePredictions, indexMap, numOfPDFeaturs);
    }

}
