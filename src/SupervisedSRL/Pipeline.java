package SupervisedSRL;

import ml.AveragedPerceptron;
import Sentence.Sentence;
import util.IO;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;


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

        //training AI and AC model
        String modelDir = args[2];
        int aiMaxBeamSize = Integer.parseInt(args[3]);
        int acMaxBeamSize = Integer.parseInt(args[4]);
        int numOfTrainingIterations = 5;
        int numOfFeatures = 188;
        String aiModelPath = Train.trainAI(trainSentencesInCONLLFormat, devSentencesInCONLLFormat,
                numOfTrainingIterations, modelDir, numOfFeatures, aiMaxBeamSize);

        String acModelPath = Train.trainAC(trainSentencesInCONLLFormat, argLabels,
                numOfTrainingIterations, modelDir, numOfFeatures);

        //AI and AC decoding
        ArgumentDecoder argumentDecoder = new ArgumentDecoder(AveragedPerceptron.loadModel(aiModelPath),
                AveragedPerceptron.loadModel(acModelPath), argLabels);


        devSize = 0;
        //making prediction over tes sentences
        System.out.println("Decoding started...");
        for (int d = 0; d < trainSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + trainSentencesInCONLLFormat.size());

            Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(d));
            argumentDecoder.predict(sentence, aiMaxBeamSize, acMaxBeamSize, numOfFeatures);
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

        HashSet<String> argLabels = Train.obtainLabels(trainSentencesInCONLLFormat);
        argLabels.add("0");

        //training AI and AC model
        String modelDir = args[2];
        int aiMaxBeamSize = Integer.parseInt(args[3]);
        int acMaxBeamSize = Integer.parseInt(args[4]);
        int numOfTrainingIterations = 5;
        int numOfFeatures = 188;

        String model =Train.train(trainSentencesInCONLLFormat,devSentencesInCONLLFormat,numOfTrainingIterations,modelDir,numOfFeatures,argLabels, acMaxBeamSize);

        //AI and AC decoding combined
        ArgumentDecoder argumentDecoder = new ArgumentDecoder(AveragedPerceptron.loadModel(model), argLabels, "combined");


        devSize = 0;
        //making prediction over tes sentences
        System.out.println("Decoding started...");
        for (int d = 0; d < trainSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + trainSentencesInCONLLFormat.size());

            Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(d));
            argumentDecoder.predict_combined(sentence, acMaxBeamSize, numOfFeatures);
        }
        System.out.println("dev size: " + devSize);
        argumentDecoder.computePrecisionRecall("AC");

    }
}
