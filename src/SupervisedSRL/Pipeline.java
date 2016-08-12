package SupervisedSRL;

import SupervisedSRL.Strcutures.ClassifierType;
import SupervisedSRL.Strcutures.CompactArray;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import ml.AveragedPerceptron;
import ml.Adam;
import de.bwaldvogel.liblinear.*;

import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.HashMap;
import java.io.File;
import java.util.zip.GZIPInputStream;

/**
 * Created by monadiab on 5/25/16.
 */
public class Pipeline {

    public static int numOfAIFeatures = 25 + 13; //+ 91 + 6; // 154 + 55 + 91 + 91 + 6; //25 + 13;
    public static int numOfACFeatures = 25 + 15; //+ 91 + 6;// + 55 + 91 + 91 + 6;
    public static int numOfPDFeatures = 9;
    public static int numOfPDTrainingIterations = 10;
    public static String unseenSymbol = ";;?;;";


    public static void main(String[] args) throws Exception {

        //getting trainJoint/test sentences
        String trainData = args[0];
        String devData = args[1];
        String modelDir = args[2];
        String outputFile = args[3];
        int aiMaxBeamSize = Integer.parseInt(args[4]);
        int acMaxBeamSize = Integer.parseInt(args[5]);
        int numOfTrainingIterations = Integer.parseInt(args[6]);
        int adamBatchSize = Integer.parseInt(args[7]);
        int learnerType = Integer.parseInt(args[8]); //1: ap 2: ll 3:adam
        boolean decodeJoint = Boolean.parseBoolean(args[9]);
        boolean decodeOnly = Boolean.parseBoolean(args[10]);
        ClassifierType classifierType = ClassifierType.AveragedPerceptron;
        switch (learnerType)
        {
            case(1):
                classifierType= ClassifierType.AveragedPerceptron;
            case(2):
                classifierType = ClassifierType.Liblinear;
            case(3):
                classifierType = ClassifierType.Adam;
        }

        //single features 25
        //p-p features 55
        //a-a feature 91
        //p-a features 154
        //p-a-a features 91
        //some msc tri-gram feature 6
        //joined features based on original paper (ai) 13
        //joined features based on original paper (ac) 15


        if (!decodeOnly) {
            Train train = new Train();
            String[] modelPaths = new String[4];
            if (decodeJoint) {
                //joint decoding
                if (classifierType == ClassifierType.AveragedPerceptron) {
                    modelPaths[0] = train.trainJoint(trainData, devData, numOfTrainingIterations, modelDir, outputFile, numOfACFeatures, numOfPDFeatures, acMaxBeamSize);
                    ModelInfo modelInfo = new ModelInfo(modelPaths[0]);
                    IndexMap indexMap = modelInfo.getIndexMap();
                    AveragedPerceptron classifier = modelInfo.getClassifier();
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, devData, classifier.getLabelMap(),
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, null, ClassifierType.AveragedPerceptron);

                    Evaluation.evaluate(outputFile, devData, indexMap, modelInfo.getClassifier().getReverseLabelMap());
                } else if (classifierType == ClassifierType.Liblinear) {
                    modelPaths = train.trainJointLiblinear(trainData, devData, numOfTrainingIterations, modelDir,
                            numOfACFeatures, numOfPDFeatures);
                    ModelInfo modelInfo = new ModelInfo(modelPaths[0], modelPaths[1], ClassifierType.Liblinear);
                    IndexMap indexMap = modelInfo.getIndexMap();
                    Model classifier = modelInfo.getClassifierLiblinear();
                    HashMap<Object, Integer>[] featDict = modelInfo.getFeatDict();
                    HashMap<String, Integer> labelDict = modelInfo.getLabelDict();
                    String[] labelMap = new String[labelDict.size()];
                    for (String label : labelDict.keySet())
                        labelMap[labelDict.get(label)] = label;
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, devData, labelMap,
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, featDict, ClassifierType.Liblinear);

                    Evaluation.evaluate(outputFile, devData, indexMap, labelDict);
                }else if (classifierType == ClassifierType.Adam) {
                    modelPaths = train.trainJointAdam(trainData, devData, numOfTrainingIterations, modelDir,
                            numOfACFeatures, numOfPDFeatures, adamBatchSize, acMaxBeamSize);
                    ModelInfo modelInfo = new ModelInfo(modelPaths[0], modelPaths[1], ClassifierType.Adam);
                    IndexMap indexMap = modelInfo.getIndexMap();
                    Adam classifier = modelInfo.getClassifierAdam();
                    HashMap<Object, Integer>[] featDict = modelInfo.getFeatDict();
                    HashMap<String, Integer> labelDict = modelInfo.getLabelDict();
                    String[] labelMap = new String[labelDict.size()];
                    for (String label : labelDict.keySet())
                        labelMap[labelDict.get(label)] = label;
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, devData, labelMap,
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, featDict, ClassifierType.Adam);

                    Evaluation.evaluate(outputFile, devData, indexMap, labelDict);
                }
            } else {
                //stacked decoding
                if (classifierType == ClassifierType.AveragedPerceptron) {

                    modelPaths = train.train(trainData, devData, numOfTrainingIterations, modelDir,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiMaxBeamSize, acMaxBeamSize, adamBatchSize,
                            ClassifierType.AveragedPerceptron);

                    ModelInfo aiModelInfo = new ModelInfo(modelPaths[0]);
                    IndexMap indexMap = aiModelInfo.getIndexMap();
                    AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
                    AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelPaths[2]);
                    System.out.println("<><><><><><><><><><><><><><><>beam "+acMaxBeamSize);
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            aiModelInfo.getIndexMap(),
                            devData, acClassifier.getLabelMap(),
                            aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acClassifier.getReverseLabelMap());
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);

                    System.out.println("<><><><><><><><><><><><><><><>beam 1");
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            aiModelInfo.getIndexMap(),
                            devData, acClassifier.getLabelMap(),
                            1, 1, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron);
                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);

                    System.out.println("<><><><><><><><><><><><><><><>beam 8");
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            aiModelInfo.getIndexMap(),
                            devData, acClassifier.getLabelMap(),
                            8, 8, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron);
                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);

                }else if (classifierType == ClassifierType.Liblinear) {
                    modelPaths = train.train(trainData, devData, numOfTrainingIterations, modelDir,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiMaxBeamSize, acMaxBeamSize, adamBatchSize,
                            ClassifierType.Liblinear);

                    ModelInfo aiModelInfo = new ModelInfo(modelPaths[0], modelPaths[1], ClassifierType.Liblinear);
                    ModelInfo acModelInfo = new ModelInfo(modelPaths[2], modelPaths[3], ClassifierType.Liblinear);
                    Model aiClassifier = aiModelInfo.getClassifierLiblinear();
                    IndexMap indexMap= aiModelInfo.getIndexMap();
                    HashMap<Object, Integer>[] aiFeatDict = aiModelInfo.getFeatDict();
                    Model acClassifier= acModelInfo.getClassifierLiblinear();
                    HashMap<Object, Integer>[] acFeatDict = acModelInfo.getFeatDict();
                    HashMap<String, Integer> acLabelDict = acModelInfo.getLabelDict();
                    String[] acLabelMap = new String[acLabelDict.size()];
                    for (String label: acLabelDict.keySet())
                        acLabelMap[acLabelDict.get(label)]= label;

                    System.out.println("<><><><><><><><><><><><><><><>beam "+acMaxBeamSize);
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                        indexMap, devData, acLabelMap, aiMaxBeamSize, acMaxBeamSize,
                        numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                        modelDir, outputFile, aiFeatDict, acFeatDict, ClassifierType.Liblinear);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acLabelDict);
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);

                    System.out.println("<><><><><><><><><><><><><><><>beam 1");
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            indexMap, devData, acLabelMap, 1, 1,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, aiFeatDict, acFeatDict, ClassifierType.Liblinear);

                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);

                    System.out.println("<><><><><><><><><><><><><><><>beam 8");
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            indexMap, devData, acLabelMap, 8, 8,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, aiFeatDict, acFeatDict, ClassifierType.Liblinear);

                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
                }else if (classifierType == ClassifierType.Adam)
                {
                    modelPaths = train.train(trainData, devData, numOfTrainingIterations, modelDir,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            aiMaxBeamSize, acMaxBeamSize, adamBatchSize, ClassifierType.Adam);

                    ModelInfo aiModelInfo = new ModelInfo(modelPaths[0], modelPaths[1], ClassifierType.Adam);
                    ModelInfo acModelInfo = new ModelInfo(modelPaths[2], modelPaths[3], ClassifierType.Adam);
                    Adam aiClassifier = aiModelInfo.getClassifierAdam();
                    IndexMap indexMap= aiModelInfo.getIndexMap();
                    HashMap<Object, Integer>[] aiFeatDict = aiModelInfo.getFeatDict();
                    Adam acClassifier= acModelInfo.getClassifierAdam();
                    HashMap<Object, Integer>[] acFeatDict = acModelInfo.getFeatDict();
                    HashMap<String, Integer> acLabelDict = acModelInfo.getLabelDict();
                    String[] acLabelMap = new String[acLabelDict.size()];
                    for (String label: acLabelDict.keySet())
                        acLabelMap[acLabelDict.get(label)]= label;

                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            indexMap, devData, acLabelMap, aiMaxBeamSize, acMaxBeamSize,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, aiFeatDict, acFeatDict, ClassifierType.Adam);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acLabelDict);
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
                }
            }
        } else {
            if (decodeJoint) {
                //joint decoding
                if (classifierType == ClassifierType.AveragedPerceptron) {
                    ModelInfo modelInfo = new ModelInfo(modelDir + "/joint.model");
                    IndexMap indexMap = modelInfo.getIndexMap();
                    AveragedPerceptron classifier = modelInfo.getClassifier();
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, devData, classifier.getLabelMap(),
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, null, ClassifierType.AveragedPerceptron);

                    Evaluation.evaluate(outputFile, devData, indexMap, classifier.getReverseLabelMap());
                }else if (classifierType == ClassifierType.Liblinear)
                {
                    ModelInfo modelInfo = new ModelInfo(modelDir+"JOINT_ll.model", modelDir+"mappingDicts_ll_JOINT", ClassifierType.Liblinear);
                    IndexMap indexMap = modelInfo.getIndexMap();
                    Model classifier = modelInfo.getClassifierLiblinear();
                    HashMap<Object, Integer>[] featDict = modelInfo.getFeatDict();
                    HashMap<String, Integer> labelDict = modelInfo.getLabelDict();
                    String[] labelMap = new String[labelDict.size()];
                    for (String label: labelDict.keySet())
                        labelMap[labelDict.get(label)]= label;
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, devData, labelMap,
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, featDict, ClassifierType.Liblinear);

                    Evaluation.evaluate(outputFile, devData, indexMap,labelDict);
                }else if (classifierType == ClassifierType.Adam) {
                    ModelInfo modelInfo = new ModelInfo(modelDir+"JOINT_adam.model", modelDir+"mappingDicts_adam_JOINT", ClassifierType.Adam);
                    IndexMap indexMap = modelInfo.getIndexMap();
                    Adam classifier = modelInfo.getClassifierAdam();
                    HashMap<Object, Integer>[] featDict = modelInfo.getFeatDict();
                    HashMap<String, Integer> labelDict = modelInfo.getLabelDict();
                    String[] labelMap = new String[labelDict.size()];
                    for (String label : labelDict.keySet())
                        labelMap[labelDict.get(label)] = label;
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, devData, labelMap,
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, featDict, ClassifierType.Adam);

                    Evaluation.evaluate(outputFile, devData, indexMap, labelDict);
                }

            } else {
                //stacked decoding
                if (classifierType == ClassifierType.AveragedPerceptron) {
                    ModelInfo aiModelInfo = new ModelInfo(modelDir + "/AI.model");
                    IndexMap indexMap = aiModelInfo.getIndexMap();
                    AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
                    AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelDir + "/AC.model");
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            aiModelInfo.getIndexMap(),
                            devData, acClassifier.getLabelMap(),
                            aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acClassifier.getReverseLabelMap());
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
                }else if (classifierType== ClassifierType.Liblinear){
                    ModelInfo aiModelInfo = new ModelInfo(modelDir + "/AI_ll.model", modelDir+"/mappingDicts_ll_AI", ClassifierType.Liblinear);
                    ModelInfo acModelInfo = new ModelInfo(modelDir + "/AC_ll.model", modelDir+"/mappingDicts_ll_AC", ClassifierType.Liblinear);
                    Model aiClassifier = aiModelInfo.getClassifierLiblinear();
                    IndexMap indexMap= aiModelInfo.getIndexMap();
                    HashMap<Object, Integer>[] aiFeatDict = aiModelInfo.getFeatDict();
                    Model acClassifier= acModelInfo.getClassifierLiblinear();
                    HashMap<Object, Integer>[] acFeatDict = acModelInfo.getFeatDict();
                    HashMap<String, Integer> acLabelDict = acModelInfo.getLabelDict();
                    String[] acLabelMap = new String[acLabelDict.size()];
                    for (String label: acLabelDict.keySet())
                        acLabelMap[acLabelDict.get(label)]= label;

                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            indexMap, devData, acLabelMap, aiMaxBeamSize, acMaxBeamSize,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, aiFeatDict, acFeatDict, ClassifierType.Liblinear);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acLabelDict);
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
                }else if (classifierType == ClassifierType.Adam){
                    ModelInfo aiModelInfo = new ModelInfo(modelDir + "/AI_adam.model", modelDir+"/mappingDicts_adam_AI", ClassifierType.Adam);
                    ModelInfo acModelInfo = new ModelInfo(modelDir + "/AC_adam.model", modelDir+"/mappingDicts_adam_AC", ClassifierType.Adam);
                    Adam aiClassifier = aiModelInfo.getClassifierAdam();
                    IndexMap indexMap= aiModelInfo.getIndexMap();
                    HashMap<Object, Integer>[] aiFeatDict = aiModelInfo.getFeatDict();
                    Adam acClassifier= acModelInfo.getClassifierAdam();
                    HashMap<Object, Integer>[] acFeatDict = acModelInfo.getFeatDict();
                    HashMap<String, Integer> acLabelDict = acModelInfo.getLabelDict();
                    String[] acLabelMap = new String[acLabelDict.size()];
                    for (String label: acLabelDict.keySet())
                        acLabelMap[acLabelDict.get(label)]= label;

                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            indexMap, devData, acLabelMap, aiMaxBeamSize, acMaxBeamSize,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, aiFeatDict, acFeatDict, ClassifierType.Adam);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acLabelDict);
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
                }
            }
        }
    }
}
