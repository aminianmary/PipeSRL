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
    public static int devSize = 0;
    /*
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
         int numOfTrainingIterations = 5;
         int numOfFeatures = 188;

         String aiModelPath = Train.trainAI(trainSentencesInCONLLFormat, devSentencesInCONLLFormat, indexMap,
                 numOfTrainingIterations, modelDir, numOfFeatures, aiMaxBeamSize);

         String acModelPath = Train.trainAC(trainSentencesInCONLLFormat, argLabels, indexMap,
                 numOfTrainingIterations, modelDir, numOfFeatures);

         //AI and AC decoding
         //ArgumentDecoder argumentDecoder = new ArgumentDecoder(AveragedPerceptron.loadModel(aiModelPath),
         //        AveragedPerceptron.loadModel(acModelPath), argLabels);
         ArgumentDecoder argumentDecoder = new ArgumentDecoder(new ModelInfo(aiModelPath), new ModelInfo(acModelPath));

         devSize = 0;
         //making prediction over tes sentences
         System.out.println("Decoding started...");
         for (int d = 0; d < trainSentencesInCONLLFormat.size(); d++) {
             if (d % 1000 == 0)
                 System.out.println(d + "/" + trainSentencesInCONLLFormat.size());

             Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(d), indexMap);
             argumentDecoder.predict(sentence, indexMap, aiMaxBeamSize, acMaxBeamSize, numOfFeatures);
         }
         System.out.println("dev size: " + devSize);
         argumentDecoder.computePrecisionRecall("AC");

     }
        */

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
        ArgumentDecoder argumentDecoder = new ArgumentDecoder(new ModelInfo(model), "combined");

        devSize = 0;
        //making prediction over tes sentences
        System.out.println("Decoding started...");
        for (int d = 0; d < trainSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + trainSentencesInCONLLFormat.size());

            Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(d), indexMap);
            argumentDecoder.predict_combined(sentence, indexMap, acMaxBeamSize, numOfFeatures);
        }
        System.out.println("dev size: " + devSize);
        argumentDecoder.computePrecisionRecall("AC");
    }
    
}
