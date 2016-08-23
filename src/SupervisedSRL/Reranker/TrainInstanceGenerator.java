package SupervisedSRL.Reranker;
import java.io.*;
import java.util.*;
import java.util.concurrent.Executors;

import SupervisedSRL.Decoder;
import SupervisedSRL.Evaluation;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Pipeline;
import SupervisedSRL.Strcutures.ClassifierType;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Train;
import ml.AveragedPerceptron;
import util.IO;

/**
 * Created by Maryam Aminian on 8/19/16.
 */
public class TrainInstanceGenerator {
    ArrayList<ArrayList<String>> trainPartitions;
    int numOfPartitions;

    public static void main(String[] args) throws Exception {
        String trainFilePath = args[0];
        String clusterFilePath = args[1];
        String modelDir = args[2];
        int numOfTrainingIterations = Integer.parseInt(args[3]);
        int aiBeamSize = Integer.parseInt(args[4]);
        int acBeamSize = Integer.parseInt(args[5]);
        boolean greedy= Boolean.parseBoolean(args[6]);

        int numOfPDIterations =10;
        int numOfPDFeatures = 9;
        int numOfAIFeatures = 25 + 3+ 5+ 13;
        int numOfACFeatures = 25 + 3+ 5+ 15;

        TrainInstanceGenerator trainInstanceGenerator = new TrainInstanceGenerator(trainFilePath, 10);
        trainInstanceGenerator.buildTrainInstances(clusterFilePath,modelDir,numOfPDFeatures,
                numOfPDIterations,numOfTrainingIterations,numOfAIFeatures,numOfACFeatures,aiBeamSize, acBeamSize, greedy);

    }

    public TrainInstanceGenerator(String trainFilePath, int numOfParts) throws IOException {
        this.numOfPartitions =  numOfParts;
        ArrayList<ArrayList<String>> partitions = new ArrayList<ArrayList<String>>();
        ArrayList<String> sentencesInCoNLLFormat = IO.readCoNLLFile(trainFilePath);
        Collections.shuffle(sentencesInCoNLLFormat);
        int partitionSize = (int) Math.ceil((double) sentencesInCoNLLFormat.size() / numOfPartitions);
        int startIndex = 0;
        int endIndex = 0;
        for (int i = 0; i < numOfPartitions; i++) {
            endIndex = startIndex + partitionSize ;
            ArrayList<String> partitionSentences = new ArrayList<String>();
            if (endIndex < sentencesInCoNLLFormat.size())
                partitionSentences = new ArrayList<String>(sentencesInCoNLLFormat.subList(startIndex, endIndex));
            else
                partitionSentences = new ArrayList<String>(sentencesInCoNLLFormat.subList(startIndex, sentencesInCoNLLFormat.size()));

            partitions.add(partitionSentences);
            startIndex = endIndex;
        }
        this.trainPartitions = partitions;
    }


    public void buildTrainInstances (String clusterFile, String modelDir, int numOfPDFeatures,
                                     int numOfPDTrainingIterations, int numberOfTrainingIterations,
                                     int numOfAIFeatures, int numOfACFeatures,
                                     int aiMaxBeamSize, int acMaxBeamSize, boolean greedy) throws Exception{
        Train train = new Train();
        for (int devPartIdx =0 ; devPartIdx < numOfPartitions; devPartIdx++){
            ArrayList<String> trainSentences = new ArrayList<String>();
            ArrayList<String> devSentences = new ArrayList<String>();

            for (int partIdx =0 ; partIdx < numOfPartitions ; partIdx++){
                if (partIdx == devPartIdx)
                    devSentences = trainPartitions.get(partIdx);
                else
                    trainSentences.addAll(trainPartitions.get(partIdx));
            }
            //write dev sentences into a file (to be compatible with previous functions)
            String devDataPath = modelDir+"/reranker_dev_"+devPartIdx;
            BufferedWriter devWriter= new BufferedWriter(new OutputStreamWriter(new FileOutputStream(devDataPath)));
            for (String sentence: devSentences)
                    devWriter.write(sentence+"\n\n");
            devWriter.flush();
            devWriter.close();

            //train a PD-AI-AC modules on the train parts
            HashSet<String> argLabels = IO.obtainLabels(trainSentences);
            final IndexMap indexMap = new IndexMap(trainSentences, clusterFile);
            PD.train(trainSentences, indexMap, numOfPDTrainingIterations, modelDir, numOfPDFeatures);
            String aiModelPath = train.trainAI(trainSentences, devSentences, indexMap,
                    numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfPDFeatures, aiMaxBeamSize, greedy);
            String acModelPath = train.trainAC(trainSentences, devDataPath, argLabels, indexMap,
                    numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                    aiMaxBeamSize, acMaxBeamSize, greedy);

            ModelInfo aiModelInfo = new ModelInfo(aiModelPath);
            AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
            AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(acModelPath);

            String outputFile = modelDir+"/system_output_"+devPartIdx;
            Decoder.decode(new Decoder(aiClassifier, acClassifier),
                    indexMap, devDataPath, acClassifier.getLabelMap(),
                    aiMaxBeamSize, acMaxBeamSize, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                    modelDir, outputFile, null, null, ClassifierType.AveragedPerceptron, greedy);

            //testing AI-AC performance on this partition
            HashMap<String, Integer> reverseLabelMap = new HashMap<String, Integer>(acClassifier.getReverseLabelMap());
            reverseLabelMap.put("0", reverseLabelMap.size());
            Evaluation.evaluate(outputFile, devDataPath, indexMap, reverseLabelMap);
        }
    }


    public void extractRerankerFeatures (String modelDir) throws IOException{
        for (int devPartIdx =0; devPartIdx < numOfPartitions; devPartIdx++){

        }
    }

}