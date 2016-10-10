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
        String evalData = args[1];
        String evalDataPredicatePredictionsFilePath = args[2];
        String indexMapFilePath = args[3];
        int numOfPDFeaturs = Integer.parseInt(args[4]);
        boolean readPredictionsFromOutputFile = Boolean.parseBoolean(args[5]);


        IndexMap indexMap = IO.load(indexMapFilePath);
        ArrayList<String> trainSentences = IO.readCoNLLFile(trainData);
        ArrayList<String> goldEvalSentences = IO.readCoNLLFile(evalData);
        HashMap<Integer, String>[] evalDataPredicatePredictions = (!readPredictionsFromOutputFile) ?
                (HashMap<Integer, String>[]) IO.load(evalDataPredicatePredictionsFilePath) :
                IO.getDisambiguatedPredicatesFromOutput(evalDataPredicatePredictionsFilePath, goldEvalSentences.size());;

        Evaluation.evaluatePD(trainSentences, goldEvalSentences, evalDataPredicatePredictions, indexMap, numOfPDFeaturs);
    }

}
