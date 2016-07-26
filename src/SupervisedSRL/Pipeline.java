package SupervisedSRL;

import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.IndexMap;
import ml.AveragedPerceptron;

import java.util.HashMap;


/**
 * Created by monadiab on 5/25/16.
 */
public class Pipeline {

     public static void main(String[] args) throws Exception {

         //getting trainJoint/test sentences
         String trainData = args[0];
         String devData = args[1];
         String modelDir = args[2];
         String outputFile = args[3];
         int aiMaxBeamSize = Integer.parseInt(args[4]);
         int acMaxBeamSize = Integer.parseInt(args[5]);
         int numOfTrainingIterations = Integer.parseInt(args[6]);
         boolean decodeJoint = Boolean.parseBoolean(args[7]);
         boolean decodeOnly = Boolean.parseBoolean(args[8]);
         boolean evalOnly = Boolean.parseBoolean(args[9]);

         int numOfFeatures = 188 +55;
         int numOfPDFeatures =9;

         if (evalOnly)
         {

         }
         else {

             if (decodeOnly == false) {
                 Train train = new Train();
                 String[] modelPaths = new String[2];
                 if (decodeJoint == true) {
                     //joint decoding
                     modelPaths[0] = train.trainJoint(trainData, devData, numOfTrainingIterations, modelDir, numOfFeatures, numOfPDFeatures, aiMaxBeamSize);
                     ModelInfo modelInfo = new ModelInfo(modelPaths[0]);
                     IndexMap indexMap = modelInfo.getIndexMap();
                     Decoder.decode(new Decoder(modelInfo.getClassifier(), "joint"),
                             indexMap,
                             devData, modelInfo.getClassifier().getLabelMap(),
                             aiMaxBeamSize, numOfFeatures, numOfPDFeatures, modelDir, outputFile);

                     Evaluation.evaluate(outputFile, devData, indexMap, modelInfo.getClassifier().getReverseLabelMap());

                 } else {
                     //stacked decoding
                     modelPaths = train.train(trainData, devData, numOfTrainingIterations, modelDir, numOfFeatures, numOfFeatures,numOfPDFeatures, aiMaxBeamSize, acMaxBeamSize);
                     ModelInfo aiModelInfo = new ModelInfo(modelPaths[0]);
                     IndexMap indexMap = aiModelInfo.getIndexMap();
                     AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
                     AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelPaths[1]);
                     Decoder.decode(new Decoder(aiClassifier, acClassifier),
                             aiModelInfo.getIndexMap(),
                             devData, acClassifier.getLabelMap(),
                             aiMaxBeamSize, acMaxBeamSize, numOfFeatures ,numOfPDFeatures,
                             modelDir, outputFile);

                     HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acClassifier.getReverseLabelMap());
                     reverseLabelMap.put("0", reverseLabelMap.size());
                     Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
                 }
             } else if (decodeOnly == true) {
                 if (decodeJoint == true) {
                     //joint decoding
                     ModelInfo modelInfo = new ModelInfo(modelDir + "/joint.model");
                     IndexMap indexMap = modelInfo.getIndexMap();
                     Decoder.decode(new Decoder(modelInfo.getClassifier(), "joint"),
                             indexMap,
                             devData, modelInfo.getClassifier().getLabelMap(),
                             aiMaxBeamSize, numOfFeatures, numOfPDFeatures, modelDir, outputFile);

                     Evaluation.evaluate(outputFile, devData, indexMap, modelInfo.getClassifier().getReverseLabelMap());
                 } else {
                     //stacked decoding
                     ModelInfo aiModelInfo = new ModelInfo(modelDir + "/AI.model");
                     IndexMap indexMap = aiModelInfo.getIndexMap();
                     AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
                     AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelDir + "/AC.model");
                     Decoder.decode(new Decoder(aiClassifier, acClassifier),
                             aiModelInfo.getIndexMap(),
                             devData, acClassifier.getLabelMap(),
                             aiMaxBeamSize, acMaxBeamSize,
                             numOfFeatures, modelDir, outputFile);

                     HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acClassifier.getReverseLabelMap());
                     reverseLabelMap.put("0", reverseLabelMap.size());
                     Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
                 }
             }
         }
     }
}
