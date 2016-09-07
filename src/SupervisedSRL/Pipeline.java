package SupervisedSRL;

import SupervisedSRL.Strcutures.ClassifierType;
import SupervisedSRL.Strcutures.ClusterMap;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import de.bwaldvogel.liblinear.Model;
import ml.Adam;
import ml.AveragedPerceptron;

import java.util.HashMap;

/**
 * Created by monadiab on 5/25/16.
 */
public class Pipeline {

    //single features 25 + 3 (predicate cluster features) + 5(argument cluster features)
    //p-p features 55
    //a-a feature 91
    //p-a features 154
    //p-a-a features 91
    //some msc tri-gram feature 6
    //joined features based on original paper (ai) 13
    //joined features based on original paper (ac) 15
    //predicate cluster features 3
    //argument cluster features 5

    public static int numOfAIFeatures = 25 + 3 + 5 + 154 + 91 + 6;
    public static int numOfACFeatures = 25 + 3 + 5 + 154 + 91 + 6;
    public static int numOfPDFeatures = 9;
    public static int numOfPDTrainingIterations = 10;
    public static String unseenSymbol = ";;?;;";


    public static void main(String[] args) throws Exception {
        //getting trainJoint/test sentences
        String trainData = args[0];
        String devData = args[1];
        String clusterFile = args[2];
        String modelDir = args[3];
        String outputFile = args[4];
        int aiMaxBeamSize = Integer.parseInt(args[5]);
        int acMaxBeamSize = Integer.parseInt(args[6]);
        int numOfTrainingIterations = Integer.parseInt(args[7]);
        int adamBatchSize = Integer.parseInt(args[8]);
        int learnerType = Integer.parseInt(args[9]); //1: ap 2: ll 3:adam
        double adamLearningRate = Double.parseDouble(args[10]);
        boolean decodeJoint = Boolean.parseBoolean(args[11]);
        boolean decodeOnly = Boolean.parseBoolean(args[12]);
        boolean greedy = Boolean.parseBoolean(args[13]);
        int numOfThreads = Integer.parseInt(args[14]);
        ClassifierType classifierType = ClassifierType.AveragedPerceptron;
        switch (learnerType) {
            case (1):
                classifierType = ClassifierType.AveragedPerceptron;
                break;
            case (2):
                classifierType = ClassifierType.Liblinear;
                break;
            case (3):
                classifierType = ClassifierType.Adam;
        }

        if (!decodeOnly) {
            String[] modelPaths = new String[4];
            if (decodeJoint) {
                //joint decoding
                if (classifierType == ClassifierType.AveragedPerceptron) {
                    modelPaths[0] = Train.trainJoint(trainData, devData, clusterFile, numOfTrainingIterations, modelDir, outputFile, numOfACFeatures, numOfPDFeatures, acMaxBeamSize, greedy);
                    ModelInfo modelInfo = new ModelInfo(modelPaths[0]);
                    IndexMap indexMap = modelInfo.getIndexMap();
                    ClusterMap clusterMap = modelInfo.getClusterMap();
                    AveragedPerceptron classifier = modelInfo.getClassifier();
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, clusterMap, devData, classifier.getLabelMap(),
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, null, ClassifierType.AveragedPerceptron, greedy);

                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, modelInfo.getClassifier().getReverseLabelMap());
                } else if (classifierType == ClassifierType.Liblinear) {
                    modelPaths = Train.trainJointLiblinear(trainData, devData, clusterFile, numOfTrainingIterations, modelDir,
                            numOfACFeatures, numOfPDFeatures);
                    ModelInfo modelInfo = new ModelInfo(modelPaths[0], modelPaths[1], ClassifierType.Liblinear);
                    IndexMap indexMap = modelInfo.getIndexMap();
                    ClusterMap clusterMap = modelInfo.getClusterMap();
                    Model classifier = modelInfo.getClassifierLiblinear();
                    HashMap<Object, Integer>[] featDict = modelInfo.getFeatDict();
                    HashMap<String, Integer> labelDict = modelInfo.getLabelDict();
                    String[] labelMap = new String[labelDict.size()];
                    for (String label : labelDict.keySet())
                        labelMap[labelDict.get(label)] = label;
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, clusterMap, devData, labelMap,
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, featDict, ClassifierType.Liblinear, greedy);

                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, labelDict);
                } else if (classifierType == ClassifierType.Adam) {
                    modelPaths = Train.trainJointAdam(trainData, devData, clusterFile, numOfTrainingIterations, modelDir,
                            numOfACFeatures, numOfPDFeatures, adamBatchSize, acMaxBeamSize, adamLearningRate,
                            greedy, numOfThreads);
                    ModelInfo modelInfo = new ModelInfo(modelPaths[0], modelPaths[1], ClassifierType.Adam);
                    IndexMap indexMap = modelInfo.getIndexMap();
                    ClusterMap clusterMap = modelInfo.getClusterMap();
                    Adam classifier = modelInfo.getClassifierAdam();
                    HashMap<Object, Integer>[] featDict = modelInfo.getFeatDict();
                    String[] labelMap = classifier.getLabelMap();
                    HashMap<String, Integer> reverseLabelMap = classifier.getReverseLabelMap();
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, clusterMap, devData, labelMap,
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, featDict, ClassifierType.Adam, greedy);

                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, reverseLabelMap);
                    classifier.shutDownLiveThreads();
                }
            } else {
                //stacked decoding
                if (classifierType == ClassifierType.AveragedPerceptron) {
                    modelPaths = Train.train(trainData, devData, clusterFile, numOfTrainingIterations, modelDir,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiMaxBeamSize, acMaxBeamSize, adamBatchSize, adamLearningRate,
                            ClassifierType.AveragedPerceptron, greedy, numOfThreads);

                    ModelInfo aiModelInfo = new ModelInfo(modelPaths[0]);
                    IndexMap indexMap = aiModelInfo.getIndexMap();
                    ClusterMap clusterMap = aiModelInfo.getClusterMap();
                    AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
                    AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelPaths[2]);
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            aiModelInfo.getIndexMap(), clusterMap,
                            devData, acClassifier.getLabelMap(),
                            aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron, greedy);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acClassifier.getReverseLabelMap());
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, reverseLabelMap);

                } else if (classifierType == ClassifierType.Liblinear) {
                    modelPaths = Train.train(trainData, devData, clusterFile, numOfTrainingIterations, modelDir,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiMaxBeamSize, acMaxBeamSize, adamBatchSize, adamLearningRate,
                            ClassifierType.Liblinear, greedy, numOfThreads);

                    ModelInfo aiModelInfo = new ModelInfo(modelPaths[0], modelPaths[1], ClassifierType.Liblinear);
                    ModelInfo acModelInfo = new ModelInfo(modelPaths[2], modelPaths[3], ClassifierType.Liblinear);
                    Model aiClassifier = aiModelInfo.getClassifierLiblinear();
                    IndexMap indexMap = aiModelInfo.getIndexMap();
                    ClusterMap clusterMap = aiModelInfo.getClusterMap();
                    HashMap<Object, Integer>[] aiFeatDict = aiModelInfo.getFeatDict();
                    Model acClassifier = acModelInfo.getClassifierLiblinear();
                    HashMap<Object, Integer>[] acFeatDict = acModelInfo.getFeatDict();
                    HashMap<String, Integer> acLabelDict = acModelInfo.getLabelDict();
                    String[] acLabelMap = new String[acLabelDict.size()];
                    for (String label : acLabelDict.keySet())
                        acLabelMap[acLabelDict.get(label)] = label;

                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            indexMap, clusterMap, devData, acLabelMap, aiMaxBeamSize, acMaxBeamSize,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, aiFeatDict, acFeatDict, ClassifierType.Liblinear, greedy);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acLabelDict);
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, reverseLabelMap);

                } else if (classifierType == ClassifierType.Adam) {
                    modelPaths = Train.train(trainData, devData, clusterFile, numOfTrainingIterations, modelDir,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            aiMaxBeamSize, acMaxBeamSize, adamBatchSize, adamLearningRate, ClassifierType.Adam, greedy, numOfThreads);

                    ModelInfo aiModelInfo = new ModelInfo(modelPaths[0], modelPaths[1], ClassifierType.Adam);
                    ModelInfo acModelInfo = new ModelInfo(modelPaths[2], modelPaths[3], ClassifierType.Adam);
                    Adam aiClassifier = aiModelInfo.getClassifierAdam();
                    IndexMap indexMap = aiModelInfo.getIndexMap();
                    ClusterMap clusterMap = aiModelInfo.getClusterMap();
                    HashMap<Object, Integer>[] aiFeatDict = aiModelInfo.getFeatDict();
                    Adam acClassifier = acModelInfo.getClassifierAdam();
                    HashMap<Object, Integer>[] acFeatDict = acModelInfo.getFeatDict();
                    String[] acLabelMap = acClassifier.getLabelMap();
                    HashMap<String, Integer> acReverseLabelMap = acClassifier.getReverseLabelMap();

                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            indexMap, clusterMap, devData, acLabelMap, aiMaxBeamSize, acMaxBeamSize,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, aiFeatDict, acFeatDict, ClassifierType.Adam, greedy);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acReverseLabelMap);
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, reverseLabelMap);
                    aiClassifier.shutDownLiveThreads();
                    acClassifier.shutDownLiveThreads();
                }
            }
        } else {
            if (decodeJoint) {
                //joint decoding
                if (classifierType == ClassifierType.AveragedPerceptron) {
                    ModelInfo modelInfo = new ModelInfo(modelDir + "/joint.model");
                    IndexMap indexMap = modelInfo.getIndexMap();
                    ClusterMap clusterMap = modelInfo.getClusterMap();
                    AveragedPerceptron classifier = modelInfo.getClassifier();
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, clusterMap, devData, classifier.getLabelMap(),
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, null, ClassifierType.AveragedPerceptron, greedy);

                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, classifier.getReverseLabelMap());
                } else if (classifierType == ClassifierType.Liblinear) {
                    ModelInfo modelInfo = new ModelInfo(modelDir + "JOINT_ll.model", modelDir + "mappingDicts_ll_JOINT", ClassifierType.Liblinear);
                    IndexMap indexMap = modelInfo.getIndexMap();
                    ClusterMap clusterMap = modelInfo.getClusterMap();
                    Model classifier = modelInfo.getClassifierLiblinear();
                    HashMap<Object, Integer>[] featDict = modelInfo.getFeatDict();
                    HashMap<String, Integer> labelDict = modelInfo.getLabelDict();
                    String[] labelMap = new String[labelDict.size()];
                    for (String label : labelDict.keySet())
                        labelMap[labelDict.get(label)] = label;
                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, clusterMap, devData, labelMap,
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, featDict, ClassifierType.Liblinear, greedy);

                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, labelDict);
                } else if (classifierType == ClassifierType.Adam) {
                    ModelInfo modelInfo = new ModelInfo(modelDir + "JOINT_adam.model", modelDir + "mappingDicts_adam_JOINT", ClassifierType.Adam);
                    IndexMap indexMap = modelInfo.getIndexMap();
                    ClusterMap clusterMap = modelInfo.getClusterMap();
                    Adam classifier = modelInfo.getClassifierAdam();
                    HashMap<Object, Integer>[] featDict = modelInfo.getFeatDict();
                    String[] labelMap = classifier.getLabelMap();
                    HashMap<String, Integer> reverseLabelMap = classifier.getReverseLabelMap();

                    Decoder.decode(new Decoder(classifier, "joint"),
                            indexMap, clusterMap, devData, labelMap,
                            acMaxBeamSize, numOfACFeatures, numOfPDFeatures, modelDir, outputFile, featDict, ClassifierType.Adam, greedy);

                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, reverseLabelMap);
                    classifier.shutDownLiveThreads();
                }

            } else {
                //stacked decoding
                if (classifierType == ClassifierType.AveragedPerceptron) {
                    ModelInfo aiModelInfo = new ModelInfo(modelDir + "/AI.model");
                    IndexMap indexMap = aiModelInfo.getIndexMap();
                    ClusterMap clusterMap = aiModelInfo.getClusterMap();
                    AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
                    AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelDir + "/AC.model");
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            indexMap, clusterMap,
                            devData, acClassifier.getLabelMap(),
                            aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron, greedy);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acClassifier.getReverseLabelMap());
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, reverseLabelMap);
                } else if (classifierType == ClassifierType.Liblinear) {
                    ModelInfo aiModelInfo = new ModelInfo(modelDir + "/AI_ll.model", modelDir + "/mappingDicts_ll_AI", ClassifierType.Liblinear);
                    ModelInfo acModelInfo = new ModelInfo(modelDir + "/AC_ll.model", modelDir + "/mappingDicts_ll_AC", ClassifierType.Liblinear);
                    Model aiClassifier = aiModelInfo.getClassifierLiblinear();
                    IndexMap indexMap = aiModelInfo.getIndexMap();
                    ClusterMap clusterMap = aiModelInfo.getClusterMap();
                    HashMap<Object, Integer>[] aiFeatDict = aiModelInfo.getFeatDict();
                    Model acClassifier = acModelInfo.getClassifierLiblinear();
                    HashMap<Object, Integer>[] acFeatDict = acModelInfo.getFeatDict();
                    HashMap<String, Integer> acLabelDict = acModelInfo.getLabelDict();
                    String[] acLabelMap = new String[acLabelDict.size()];
                    for (String label : acLabelDict.keySet())
                        acLabelMap[acLabelDict.get(label)] = label;

                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            indexMap, clusterMap, devData, acLabelMap, aiMaxBeamSize, acMaxBeamSize,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, aiFeatDict, acFeatDict, ClassifierType.Liblinear, greedy);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acLabelDict);
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, reverseLabelMap);
                } else if (classifierType == ClassifierType.Adam) {
                    ModelInfo aiModelInfo = new ModelInfo(modelDir + "/AI_adam.model", modelDir + "/mappingDicts_adam_AI", ClassifierType.Adam);
                    ModelInfo acModelInfo = new ModelInfo(modelDir + "/AC_adam.model", modelDir + "/mappingDicts_adam_AC", ClassifierType.Adam);
                    Adam aiClassifier = aiModelInfo.getClassifierAdam();
                    IndexMap indexMap = aiModelInfo.getIndexMap();
                    ClusterMap clusterMap = aiModelInfo.getClusterMap();
                    HashMap<Object, Integer>[] aiFeatDict = aiModelInfo.getFeatDict();
                    Adam acClassifier = acModelInfo.getClassifierAdam();
                    HashMap<Object, Integer>[] acFeatDict = acModelInfo.getFeatDict();
                    String[] acLabelMap = acClassifier.getLabelMap();
                    HashMap<String, Integer> acReverseLabelMap = acClassifier.getReverseLabelMap();
                    Decoder.decode(new Decoder(aiClassifier, acClassifier),
                            indexMap, clusterMap, devData, acLabelMap, aiMaxBeamSize, acMaxBeamSize,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                            modelDir, outputFile, aiFeatDict, acFeatDict, ClassifierType.Adam, greedy);

                    HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acReverseLabelMap);
                    reverseLabelMap.put("0", reverseLabelMap.size());
                    Evaluation.evaluate(outputFile, devData, indexMap, clusterMap, reverseLabelMap);
                    aiClassifier.shutDownLiveThreads();
                    acClassifier.shutDownLiveThreads();
                }
            }
        }
    }
}
