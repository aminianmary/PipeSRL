package SupervisedSRL;

import SupervisedSRL.Strcutures.ModelInfo;
import ml.AveragedPerceptron;


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
         int numOfTrainingIterations = Integer.parseInt(args[6]);
         String outputFile = args[7];
         int numOfFeatures = 188;

         Train train= new Train();
         String[] modelPaths = new String[2];
         if (trainJoint==true) {
             //joint decoding
             modelPaths[0] =train.trainJoint(trainData, devData, numOfTrainingIterations, modelDir, numOfFeatures, aiMaxBeamSize);
             ModelInfo modelInfo = new ModelInfo(modelPaths[0]);

             Decoder.decode(new Decoder(modelInfo.getClassifier(), "joint"), modelInfo.getIndexMap(),
                     devData, aiMaxBeamSize, numOfFeatures, outputFile);

         }
         else {
             //stacked decoding
             modelPaths= train.train(trainData, numOfTrainingIterations, modelDir, numOfFeatures, numOfFeatures);
             ModelInfo aiModelInfo = new ModelInfo(modelPaths[0]);

             AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
             AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelPaths[1]);
             Decoder.decode(new Decoder(aiClassifier, acClassifier),
                     aiModelInfo.getIndexMap(),
                     devData, acClassifier.getLabelMap(),
                     aiMaxBeamSize, acMaxBeamSize,
                     numOfFeatures, modelDir, outputFile);
         }


         /*
         String aiModelPath = args[0];
         String acModelPath = args[1];
         String devData = args[2];
         int aiMaxBeamSize = Integer.parseInt(args[3]);
         int acMaxBeamSize = Integer.parseInt(args[4]);
         int numOfFeatures = 188;

         ModelInfo aiModelInfo = new ModelInfo(aiModelPath);
         AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
         AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(acModelPath);
         ArgumentDecoder.decode(new ArgumentDecoder(aiClassifier, acClassifier),
                 aiModelInfo.getIndexMap(), devData, aiMaxBeamSize, acMaxBeamSize, numOfFeatures);
         */
     }
}
