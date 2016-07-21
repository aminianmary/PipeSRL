package SupervisedSRL;

import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.IndexMap;
import ml.AveragedPerceptron;

import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
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
         //todo change this numbers
         int numOfAINominalFeatures = 31;
         int numOfAIVerbalFeatures = 31;
         int numOfACNomialFeatures = 44;
         int numOfACVerbalFeatures = 76;
         int numOfPDFeatures =8;


         if (decodeOnly==false)
         {
             Train train= new Train();
             String[] modelPaths = new String[2];
             if (decodeJoint==true) {
                 /*
                 //joint decoding
                 modelPaths[0] =train.trainJoint(trainData, devData, numOfTrainingIterations, modelDir, numOfFeatures, aiMaxBeamSize);
                 ModelInfo modelInfo = new ModelInfo(modelPaths[0]);
                 IndexMap indexMap = modelInfo.getIndexMap();
                 Decoder.decode(new Decoder(modelInfo.getClassifier(), "joint"),
                         indexMap,
                         devData, modelInfo.getClassifier().getLabelMap(),
                         aiMaxBeamSize, numOfFeatures, modelDir, outputFile);

                 Set<String> argLabels = acNominalClassifier.getReverseLabelMap().keySet();
                 argLabels.addAll(acVerbalClassifier.getReverseLabelMap().keySet());
                 argLabels.add("0");

                 Evaluation.evaluate(outputFile, devData, indexMap, modelInfo.getClassifier().getReverseLabelMap());
                 */

             }
             else {
                 //stacked decoding
                 modelPaths= train.train(trainData, numOfTrainingIterations, modelDir, numOfPDFeatures,
                         numOfAINominalFeatures, numOfAIVerbalFeatures, numOfACNomialFeatures, numOfACVerbalFeatures);

                 String aiModelPath = modelPaths[0];
                 String acModelPath = modelPaths[1];

                 ModelInfo aiModelInfo = new ModelInfo(aiModelPath, true);
                 IndexMap indexMap = aiModelInfo.getIndexMap();
                 AveragedPerceptron aiNominalClassifier = aiModelInfo.getNominalClassifier();
                 AveragedPerceptron aiVerbalClassifier = aiModelInfo.getVerbalClassifier();

                 ModelInfo acModelInfo = new ModelInfo(acModelPath, false);
                 AveragedPerceptron acNominalClassifier = acModelInfo.getNominalClassifier();
                 AveragedPerceptron acVerbalClassifier = acModelInfo.getVerbalClassifier();

                 Decoder.decode(new Decoder(aiNominalClassifier, aiVerbalClassifier, acNominalClassifier, acVerbalClassifier),
                         aiModelInfo.getIndexMap(),
                         trainData, aiMaxBeamSize, acMaxBeamSize, numOfPDFeatures,
                         numOfAINominalFeatures, numOfAIVerbalFeatures, numOfACNomialFeatures, numOfACVerbalFeatures, modelDir, outputFile);

                 HashSet<String> argLabels = new HashSet<String>(acNominalClassifier.getReverseLabelMap().keySet());
                 for (String label:acVerbalClassifier.getReverseLabelMap().keySet())
                    argLabels.add(label);
                 argLabels.add("0");

                 Evaluation.evaluate(outputFile, trainData, indexMap, argLabels);
             }
         }
         else if (decodeOnly==true)
         {
             if (decodeJoint==true) {
                 /*
                 //joint decoding
                 ModelInfo modelInfo = new ModelInfo(modelDir+"/joint.model");
                 IndexMap indexMap = modelInfo.getIndexMap();
                 Decoder.decode(new Decoder(modelInfo.getClassifier(), "joint"),
                         indexMap,
                         devData, modelInfo.getClassifier().getLabelMap(),
                         aiMaxBeamSize, numOfFeatures, modelDir, outputFile);

                 Evaluation.evaluate(outputFile, devData, indexMap, modelInfo.getClassifier().getReverseLabelMap());
                 */
             }
             else {
                 //stacked decoding
                 ModelInfo aiModelInfo = new ModelInfo(modelDir + "/AI.model", true);
                 IndexMap indexMap = aiModelInfo.getIndexMap();
                 AveragedPerceptron aiNominalClassifier = aiModelInfo.getNominalClassifier();
                 AveragedPerceptron aiVerbalClassifier = aiModelInfo.getVerbalClassifier();

                 ModelInfo acModelInfo = new ModelInfo(modelDir + "/AC.model", false);
                 AveragedPerceptron acNominalClassifier = acModelInfo.getNominalClassifier();
                 AveragedPerceptron acVerbalClassifier = acModelInfo.getVerbalClassifier();

                 Decoder.decode(new Decoder(aiNominalClassifier, aiVerbalClassifier, acNominalClassifier, acVerbalClassifier),
                         aiModelInfo.getIndexMap(),
                         devData, aiMaxBeamSize, acMaxBeamSize, numOfPDFeatures,
                         numOfAINominalFeatures, numOfAIVerbalFeatures, numOfACNomialFeatures, numOfACVerbalFeatures, modelDir, outputFile);

                 Set<String> argLabels = acNominalClassifier.getReverseLabelMap().keySet();
                 argLabels.addAll(acVerbalClassifier.getReverseLabelMap().keySet());
                 argLabels.add("0");

                 Evaluation.evaluate(outputFile, devData, indexMap, argLabels);

             }
         }
     }
}
