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

         //getting trainJoint/test sentences
         String trainData = args[0];
         String devData = args[1];
         String modelDir = args[2];
         boolean trainJoint = Boolean.parseBoolean(args[3]);
         int aiMaxBeamSize = Integer.parseInt(args[4]);
         int acMaxBeamSize = Integer.parseInt(args[5]);
         int numOfTrainingIterations = Integer.parseInt(args[6]);;
         int numOfFeatures = 188;

         Train train= new Train();
         String[] modelPaths = new String[2];
         if (trainJoint==true) {
             //joint decoding
             modelPaths[0] =train.trainJoint(trainData, devData, numOfTrainingIterations, modelDir, numOfFeatures, aiMaxBeamSize);
             ModelInfo modelInfo = new ModelInfo(modelPaths[0]);

             ArgumentDecoder.decode(new ArgumentDecoder(modelInfo.getClassifier(), "joint"), modelInfo.getIndexMap(),
                     devData, aiMaxBeamSize, numOfFeatures);

         }
         else {
             //stacked decoding
             modelPaths= train.train(trainData, numOfTrainingIterations, modelDir, numOfFeatures, numOfFeatures);
             ModelInfo aiModelInfo = new ModelInfo(modelPaths[0]);

             AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
             AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelPaths[1]);
             ArgumentDecoder.decode(new ArgumentDecoder(aiClassifier, acClassifier),
                     aiModelInfo.getIndexMap(), devData, aiMaxBeamSize, acMaxBeamSize, numOfFeatures);
         }

     }
}
