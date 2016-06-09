package SupervisedSRL;

import ml.AveragedPerceptron;
import Sentence.Sentence;
import util.IO;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;


/**
 * Created by monadiab on 5/25/16.
 */
public class Pipeline {
    public static int testSize = 0;

    public static void main(String[] args) throws Exception {

        //getting train/test sentences
        String inputFile = args[0];
        ArrayList<String> sentencesInCONLLFormat = IO.readCoNLLFile(inputFile);
        int totalNumOfSentences = sentencesInCONLLFormat.size();
        int trainSize = (int) Math.floor(0.8 * totalNumOfSentences);

        List<String> trainSentencesInCONLLFormat = sentencesInCONLLFormat.subList(0, trainSize);
        List<String> testSentencesInCONLLFormat = sentencesInCONLLFormat.subList(trainSize, totalNumOfSentences);

        //training AI and AC model
        String modelDir = args[1];
        int numOfTrainingIterations = 5;
        String aiModelPath = Train.trainAI(trainSentencesInCONLLFormat, numOfTrainingIterations, modelDir);
     //   Object[] acModelObj = Train.trainAC(trainSentencesInCONLLFormat, numOfTrainingIterations, modelDir);

       // String acModelPath = (String) acModelObj[0];
     //   HashSet<String> acLabelSet = (HashSet<String>) acModelObj[1];

        //AI and AC decoding
        int aiMaxBeamSize = Integer.parseInt(args[2]);
        int acMaxBeamSize = Integer.parseInt(args[3]);

        ArgumentDecoder argumentDecoder = new ArgumentDecoder(AveragedPerceptron.loadModel(aiModelPath)/*,
                AveragedPerceptron.loadModel(acModelPath), acLabelSet*/);

        testSize = 0;
        //making prediction over test sentences
        System.out.println("Decoding started...");
        for (int d = 0; d < trainSentencesInCONLLFormat.size(); d++) {
            if (d%1000==0)
                System.out.println(d+"/"+trainSentencesInCONLLFormat.size());

            Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(d));
            argumentDecoder.predict(sentence, aiMaxBeamSize, acMaxBeamSize);
        }
        System.out.println("test size: "+testSize);

        argumentDecoder.computePrecisionRecall();
    }
}
