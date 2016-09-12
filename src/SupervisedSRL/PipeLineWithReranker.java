package SupervisedSRL;

import SupervisedSRL.Reranker.Decoder;
import SupervisedSRL.Reranker.RerankerInstanceGenerator;
import SupervisedSRL.Reranker.Train;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import ml.AveragedPerceptron;

import java.util.HashMap;

/**
 * Created by Maryam Aminian on 8/29/16.
 */
public class PipeLineWithReranker {

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

    public static int numOfAIFeatures = 25 + 3 + 5 + 13;
    public static int numOfACFeatures = 25 + 3 + 5 + 15;
    public static int numOfPDFeatures = 9;
    public static int numOfPDTrainingIterations = 10;

    public static void main(String[] args) throws Exception {
        String trainData = args[0];
        String devData = args[1];
        String clusterFile = args[2];
        String modelDir = args[3];
        String outputFile = args[4];
        int aiMaxBeamSize = Integer.parseInt(args[5]);
        int acMaxBeamSize = Integer.parseInt(args[6]);
        int numOfTrainingIterations = Integer.parseInt(args[7]);

        String[] modelPaths = new String[4];
            //train AI and AC model on the whole train data
            System.out.print("\n\nSTEP 1 Training PD, AI and AC Models on entire train data\n\n");
            // todo directly load pre-trained AC/AI models with indexMap
            modelPaths = SupervisedSRL.Train.train(trainData, devData, modelDir, clusterFile, numOfTrainingIterations, modelDir,
                    numOfAIFeatures, numOfACFeatures, numOfPDFeatures, aiMaxBeamSize, acMaxBeamSize);

            ModelInfo aiModelInfo = new ModelInfo(modelPaths[0]);
            IndexMap indexMap = aiModelInfo.getIndexMap();
            AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
            AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(modelPaths[2]);
            HashMap<String, Integer> globalReverseLabelMap = acClassifier.getReverseLabelMap();

            //train reranker
            //1- generate train instances
            System.out.print("\n\nSTEP 2 Training Reranker Model\n\n");
            System.out.print("\nSTEP 2.1 Generating training instances\n");
            int numOfPartitions = 5;
            String instanceFilePrefix = modelDir + "/reranker_train_instances_";
            int numOfGlobalFeatures = 1;

            // todo before instance generator; should have another func/class/etc. for feature map generation.

            // todo now all features should be from feature map wrapper
            RerankerInstanceGenerator rerankerInstanceGenerator = new RerankerInstanceGenerator(numOfPartitions,
                    modelDir, clusterFile, instanceFilePrefix, numOfPDFeatures, numOfPDTrainingIterations, numOfTrainingIterations, numOfAIFeatures,
                    numOfACFeatures, numOfGlobalFeatures, aiMaxBeamSize, acMaxBeamSize, globalReverseLabelMap);
            rerankerInstanceGenerator.buildTrainInstances(trainData);

            //2- train reranker
            System.out.print("\nSTEP 2.1 Train Reranker Model\n");
            String rerankerModelPath = Train.trainReranker(numOfPartitions, instanceFilePrefix, numOfTrainingIterations, numOfAIFeatures + numOfACFeatures + numOfGlobalFeatures, modelDir);
            AveragedPerceptron reranker = AveragedPerceptron.loadModel(rerankerModelPath);

            //decode using reranker
            System.out.print("\n\nSTEP 3 Running Decoder on Dev data (using reranker model)\n\n");
            Decoder decoder = new Decoder(aiClassifier, acClassifier, reranker, indexMap, modelDir);
            decoder.decode(devData, numOfPDFeatures, numOfAIFeatures, numOfACFeatures, aiMaxBeamSize, acMaxBeamSize, modelDir, outputFile);

            System.out.print("\n\nSTEP 4 Evaluation\n\n");
            HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(globalReverseLabelMap);
            reverseLabelMap.put("0", reverseLabelMap.size());
            Evaluation.evaluate(outputFile, devData, indexMap, reverseLabelMap);
    }

}
