package SupervisedSRL;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import ml.AveragedPerceptron;
import Sentence.Sentence;
import util.IO;

import java.io.IOException;
import java.util.*;


/**
 * Created by monadiab on 5/25/16.
 */
public class Pipeline {

     public static void main(String[] args) throws Exception {

         //getting train/test sentences
         String trainData = args[0];
         String devData = args[1];

         List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
         List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
         HashSet<String> argLabels = Train.obtainLabels(trainSentencesInCONLLFormat);

         final IndexMap indexMap = new IndexMap(trainData);

         //training AI and AC model
         String modelDir = args[2];
         int aiMaxBeamSize = Integer.parseInt(args[3]);
         int acMaxBeamSize = Integer.parseInt(args[4]);
         int numOfTrainingIterations = 1;
         int numOfFeatures = 188;

         String aiModelPath = Train.trainAI(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, indexMap,
                 numOfTrainingIterations, modelDir, numOfFeatures, aiMaxBeamSize);

         String acModelPath = Train.trainAC(trainSentencesInCONLLFormat, argLabels, indexMap,
                 numOfTrainingIterations, modelDir, numOfFeatures);

         //AI and AC decoding
         //ArgumentDecoder argumentDecoder = new ArgumentDecoder(AveragedPerceptron.loadModel(aiModelPath),
         //        AveragedPerceptron.loadModel(acModelPath), argLabels);
         System.out.println("Decoding started (on train data)...");
         ArgumentDecoder argumentDecoder = new ArgumentDecoder(AveragedPerceptron.loadModel(aiModelPath), AveragedPerceptron.loadModel(acModelPath));
         boolean decode = true;
         for (int d = 0; d < trainSentencesInCONLLFormat.size(); d++) {
             if (d % 1000 == 0)
                 System.out.println(d + "/" + trainSentencesInCONLLFormat.size());

             Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(d), indexMap, decode);
             argumentDecoder.predict(sentence, indexMap, aiMaxBeamSize, acMaxBeamSize, numOfFeatures);
         }
         argumentDecoder.computePrecisionRecall("AC");

         System.out.print("*******************************\n");

         System.out.println("Decoding started (on dev data)...");
         argumentDecoder = new ArgumentDecoder(AveragedPerceptron.loadModel(aiModelPath), AveragedPerceptron.loadModel(acModelPath));
         for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
             if (d % 1000 == 0)
                 System.out.println(d + "/" + devSentencesInCONLLFormat.size());

             Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap, decode);
             argumentDecoder.predict(sentence, indexMap, aiMaxBeamSize, acMaxBeamSize, numOfFeatures);
         }
         argumentDecoder.computePrecisionRecall("AC");

     }

    /*
    //this main function is used for ai-ac modules combined
    public static void main(String[] args) throws Exception {

        //getting train/test sentences
        String trainData = args[0];
        String devData = args[1];

        List<String> trainSentencesInCONLLFormat = IO.readCoNLLFile(trainData);
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);

        final IndexMap indexMap = new IndexMap(trainData);

        HashSet<String> argLabels = Train.obtainLabels(trainSentencesInCONLLFormat);
        argLabels.add("0");

        //training AI and AC model
        String modelDir = args[2];
        int aiMaxBeamSize = Integer.parseInt(args[3]);
        int acMaxBeamSize = Integer.parseInt(args[4]);
        int numOfTrainingIterations = 5;
        int numOfFeatures = 188;

        String model =Train.train(trainSentencesInCONLLFormat,devSentencesInCONLLFormat, indexMap,
                numOfTrainingIterations,modelDir,numOfFeatures,argLabels, acMaxBeamSize);

        //AI and AC decoding combined
        System.out.println("Decoding started (on train data)...");
        ArgumentDecoder argumentDecoder = new ArgumentDecoder(AveragedPerceptron.loadModel(model), argLabels, "combined");
        boolean decode = true;
        for (int d = 0; d < trainSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + trainSentencesInCONLLFormat.size());

            Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(d), indexMap, decode);
            argumentDecoder.predict_combined(sentence, indexMap, acMaxBeamSize, numOfFeatures);
        }
        argumentDecoder.computePrecisionRecall("AC");

        System.out.println("***********************");

        System.out.println("Decoding started (on dev data)...");
        argumentDecoder = new ArgumentDecoder(new ModelInfo(model), "combined");

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());

            Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(d), indexMap, decode);
            argumentDecoder.predict_combined(sentence, indexMap, acMaxBeamSize, numOfFeatures);
        }
        argumentDecoder.computePrecisionRecall("AC");
    }
    */
}
