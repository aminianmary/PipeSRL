package SupervisedSRL.Reranker;

import SentStructs.Argument;
import SentStructs.PA;
import SentStructs.Sentence;
import SupervisedSRL.Decoder;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.*;
import SupervisedSRL.Train;
import ml.AveragedPerceptron;
import util.IO;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by Maryam Aminian on 8/19/16.
 */
public class RerankerInstanceGenerator {
    int numOfPartitions;
    String clusterFile;
    String modelDir;
    String rerankerInstanceFilePrefix;
    int numOfPDFeatures;
    int numOfPDTrainingIterations;
    int numberOfTrainingIterations;
    int numOfAIFeatures;
    int numOfACFeatures;
    int numOfGlobalFeatures;
    int aiMaxBeamSize;
    int acMaxBeamSize;
    boolean greedy;
    //this hashMap keeps a mapping from all labels seen in our train data to integers
    //as we train several ai-ac classifiers on different partitions, a single label might end up getting different integers
    //from different classifiers and this map makes sure it won't happen
    HashMap<String, Integer> globalReverseLabelMap;

    public RerankerInstanceGenerator(int numOfParts, String modelDir, String instanceFilePrefix,
                                     int numOfPDFeatures, int numOfPDTrainingIterations, int numberOfTrainingIterations,
                                     int numOfAIFeatures, int numOfACFeatures, int numOfGlobalFeatures, int aiMaxBeamSize,
                                     int acMaxBeamSize, boolean greedy, HashMap<String, Integer> globalReverseLabelMap)
            throws IOException {
        this.numOfPartitions = numOfParts;
        this.modelDir = modelDir;
        this.rerankerInstanceFilePrefix = instanceFilePrefix;
        this.numOfPDFeatures = numOfPDFeatures;
        this.numOfPDTrainingIterations = numOfPDTrainingIterations;
        this.numberOfTrainingIterations = numberOfTrainingIterations;
        this.numOfAIFeatures = numOfAIFeatures;
        this.numOfACFeatures = numOfACFeatures;
        this.numOfGlobalFeatures = numOfGlobalFeatures;
        this.aiMaxBeamSize = aiMaxBeamSize;
        this.acMaxBeamSize = acMaxBeamSize;
        this.greedy = greedy;
        this.globalReverseLabelMap = globalReverseLabelMap;
    }

    public static HashMap<Integer, Integer> getArgLabelMap(Pair<Double, ArrayList<Integer>> aiCandid,
                                                           Pair<Double, ArrayList<Integer>> acCandid,
                                                           String[] labelMap, HashMap<String, Integer> globalReverseLabelMap) {
        HashMap<Integer, Integer> argLabelMap = new HashMap<Integer, Integer>();
        assert aiCandid.second.size() == acCandid.second.size();
        for (int i = 0; i < aiCandid.second.size(); i++) {
            int wordIdx = aiCandid.second.get(i);
            int label = globalReverseLabelMap.get(labelMap[acCandid.second.get(i)]);
            assert !argLabelMap.containsKey(wordIdx);
            argLabelMap.put(wordIdx, label);
        }
        return argLabelMap;
    }

    public static HashMap<Object, Integer>[] extractRerankerFeatures(int pIdx, String pLabel, Sentence sentence, Pair<Double, ArrayList<Integer>> aiCandid, Pair<Double,
            ArrayList<Integer>> acCandid, int numOfAIFeats, int numOfACFeats, IndexMap indexMap, String[] labelMap, HashMap<String, Integer> globalReverseLabelMap) throws Exception {
        HashMap<Integer, Integer> argMap = getArgLabelMap(aiCandid, acCandid, labelMap, globalReverseLabelMap);
        int numOfGlobalFeatures = 1;
        HashMap<Object, Integer>[] rerankerFeatureVector = new HashMap[numOfAIFeats + numOfACFeats + numOfGlobalFeatures];
        Object[] aiFeats;
        Object[] acFeats;
        Object[] globalFeats;

        for (int wordIdx = 0; wordIdx < sentence.getWords().length; wordIdx++) {
            //for each word in the sentence
            int aiLabel = (argMap.containsKey(wordIdx)) ? 1 : 0;
            int acLabel = (argMap.containsKey(wordIdx)) ? argMap.get(wordIdx) : -1;
            aiFeats = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfAIFeats, indexMap, true, aiLabel);
            acFeats = (acLabel == -1) ? null : FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfACFeats, indexMap, true, acLabel);
            addToRerankerFeats(rerankerFeatureVector, aiFeats, 0);
            addToRerankerFeats(rerankerFeatureVector, acFeats, aiFeats.length);
        }
        globalFeats = FeatureExtractor.extractGlobalFeatures(pIdx, pLabel, aiCandid, acCandid, labelMap);
        addToRerankerFeats(rerankerFeatureVector, globalFeats, numOfAIFeats + numOfACFeats);
        return rerankerFeatureVector;
    }

    public static void addToRerankerFeats(HashMap<Object, Integer>[] rerankerFeatureVector, Object[] feats, int offset) {
        if (feats == null) return;
        for (int i = 0; i < feats.length; i++) {
            if (rerankerFeatureVector[offset + i] == null)
                rerankerFeatureVector[offset + i] = new HashMap<Object, Integer>();

            if (!rerankerFeatureVector[offset + i].containsKey(feats[i]))
                rerankerFeatureVector[offset + i].put(feats[i], 1);
            else
                rerankerFeatureVector[offset + i].put(feats[i], rerankerFeatureVector[offset + i].get(feats[i]) + 1);
        }
    }

    public ArrayList<String>[] getPartitions(String trainFilePath) throws IOException {
        ArrayList<String>[] partitions = new ArrayList[numOfPartitions];
        ArrayList<String> sentencesInCoNLLFormat = IO.readCoNLLFile(trainFilePath);
        //Collections.shuffle(sentencesInCoNLLFormat);
        int partitionSize = (int) Math.ceil((double) sentencesInCoNLLFormat.size() / numOfPartitions);
        int startIndex = 0;
        int endIndex = 0;
        for (int i = 0; i < numOfPartitions; i++) {
            endIndex = startIndex + partitionSize;
            ArrayList<String> partitionSentences = new ArrayList<String>();
            if (endIndex < sentencesInCoNLLFormat.size())
                partitionSentences = new ArrayList<String>(sentencesInCoNLLFormat.subList(startIndex, endIndex));
            else
                partitionSentences = new ArrayList<String>(sentencesInCoNLLFormat.subList(startIndex, sentencesInCoNLLFormat.size()));

            partitions[i] = partitionSentences;
            startIndex = endIndex;
        }
        return partitions;
    }

    public void buildTrainInstances(String trainFilePath, ClusterMap clusterMap) throws Exception {

        ArrayList<String>[] trainParts = new ArrayList[numOfPartitions];
        trainParts = getPartitions(trainFilePath);
        ExecutorService executor = Executors.newFixedThreadPool(3);

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            executor.execute(new InstanceGenerator(trainParts, clusterMap, devPartIdx));
        }
        System.out.println("Reranker training instance generation are submitted");
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("All jobs done!");
    }

    private void writeTrainSentences(ArrayList<String>[] trainPartitions, ClusterMap clusterMap, int devPartIdx) throws Exception {
        System.out.println("generating reranker train instances for part " + devPartIdx);
        String partitionModelDir = modelDir + "/part_" + devPartIdx + "/";
        String devDataPath = partitionModelDir + "reranker_dev_" + devPartIdx;
        IO.makeDirectory(partitionModelDir);

        Object[] objs = obtainTrainDevSentences(trainPartitions, devPartIdx, devDataPath);
        ArrayList<String> trainSentences = (ArrayList<String>) objs[0];
        ArrayList<String> devSentences = (ArrayList<String>) objs[1];

        //train a PD-AI-AC modules on the train parts
        HashSet<String> argLabels = new HashSet<String>(globalReverseLabelMap.keySet());
        final IndexMap indexMap = new IndexMap(trainSentences, clusterMap, numOfAIFeatures, false);
        PD.train(trainSentences, indexMap, clusterMap, numOfPDTrainingIterations, partitionModelDir, numOfPDFeatures);
        String aiModelPath = Train.trainAI(trainSentences, devSentences, indexMap, clusterMap,
                numberOfTrainingIterations, partitionModelDir, numOfAIFeatures, numOfPDFeatures, aiMaxBeamSize, greedy);
        String acModelPath = Train.trainAC(trainSentences, devDataPath, argLabels, indexMap, clusterMap,
                numberOfTrainingIterations, partitionModelDir, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                aiMaxBeamSize, acMaxBeamSize, greedy);

        //decode on dev part
        System.out.println("Decoding started on " + devDataPath + "...\n");

        ModelInfo aiModelInfo = new ModelInfo(aiModelPath);
        AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
        AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(acModelPath);
        Decoder decoder = new Decoder(aiClassifier, acClassifier);
        String rerankerPoolsFilePath = rerankerInstanceFilePrefix + devPartIdx;
        FileOutputStream fos = new FileOutputStream(rerankerPoolsFilePath);
        ObjectOutputStream writer = new ObjectOutputStream(fos);

        //define objects used on the loops
        HashMap<Integer, HashMap<Integer, Integer>> goldMap;
        String devSentence;
        Sentence sentence;
        TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates4thisSen;
        String pLabel;
        HashMap<Integer, Integer> goldMap4ThisPredicate;

        for (int d = 0; d < devSentences.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentences.size());

            devSentence = devSentences.get(d);
            sentence = new Sentence(devSentence, indexMap, clusterMap);
            goldMap = getGoldArgLabelMap(sentence);

            predictedAIACCandidates4thisSen =
                    (TreeMap<Integer, Prediction4Reranker>) decoder.predict(sentence, indexMap, aiMaxBeamSize, acMaxBeamSize,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures, partitionModelDir,
                            null, null, greedy, true);

            //creating the pool
            for (int pIdx : predictedAIACCandidates4thisSen.keySet()) {
                pLabel = predictedAIACCandidates4thisSen.get(pIdx).getPredicateLabel();
                //to keep scores/feats of different assignments
                goldMap4ThisPredicate = goldMap.get(pIdx);

                ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = predictedAIACCandidates4thisSen.get(pIdx).getAiCandidates();
                ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = predictedAIACCandidates4thisSen.get(pIdx).getAcCandidates();
                RerankerPool rerankerPool = new RerankerPool();
                for (int i = 0; i < aiCandidates.size(); i++) {
                    Pair<Double, ArrayList<Integer>> aiCandid = aiCandidates.get(i);
                    ArrayList<Pair<Double, ArrayList<Integer>>> acCandids4thisAiCandid = acCandidates.get(i);

                    for (int j = 0; j < acCandids4thisAiCandid.size(); j++) {
                        Pair<Double, ArrayList<Integer>> acCandid = acCandids4thisAiCandid.get(j);
                        rerankerPool.addInstance(new RerankerInstanceItem(extractRerankerFeatures(pIdx, pLabel, sentence, aiCandid, acCandid,
                                numOfAIFeatures, numOfACFeatures, indexMap, acClassifier.getLabelMap(), globalReverseLabelMap), "0"), false);
                    }
                }
                //add gold assignment to the pool
                rerankerPool.addInstance(new RerankerInstanceItem(extractRerankerFeatures4GoldAssignment(pIdx, sentence, goldMap4ThisPredicate,
                        numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures, indexMap, acClassifier.getLabelMap()), "1"), true);
                writer.writeObject(rerankerPool);
            }
            writer.flush();
            System.gc();
        }
        //write dev pools
        System.out.println("Writing rerankerPools for dev part " + devPartIdx + " done!");
        writer.flush();
        writer.close();
    }

    private Object[] obtainTrainDevSentences(ArrayList<String>[] trainPartitions, int devPartIdx, String devDataPath)
            throws IOException {
        ArrayList<String> trainSentences = new ArrayList<String>();
        ArrayList<String> devSentences = new ArrayList<String>();

        for (int partIdx = 0; partIdx < numOfPartitions; partIdx++) {
            if (partIdx == devPartIdx)
                devSentences = trainPartitions[partIdx];
            else
                trainSentences.addAll(trainPartitions[partIdx]);
        }
        //write dev sentences into a file (to be compatible with previous functions)
        BufferedWriter devWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(devDataPath)));
        for (String sentence : devSentences)
            devWriter.write(sentence + "\n\n");
        devWriter.flush();
        devWriter.close();

        return new Object[]{trainSentences, devSentences};
    }

    private HashMap<Integer, HashMap<Integer, Integer>> getGoldArgLabelMap(Sentence sentence) {
        HashMap<Integer, HashMap<Integer, Integer>> goldArgLabelMap = new HashMap<Integer, HashMap<Integer, Integer>>();
        ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa : goldPAs) {
            int goldPIdx = pa.getPredicateIndex();
            ArrayList<Argument> goldArgs = pa.getArguments();
            HashMap<Integer, Integer> goldArgMap = new HashMap<Integer, Integer>();
            for (Argument arg : goldArgs)
                goldArgMap.put(arg.getIndex(), globalReverseLabelMap.get(arg.getType()));
            goldArgLabelMap.put(goldPIdx, goldArgMap);
        }
        return goldArgLabelMap;
    }

    private HashMap<Object, Integer>[] extractRerankerFeatures4GoldAssignment(int pIdx,
                                                                              Sentence sentence, HashMap<Integer, Integer> goldMap,
                                                                              int numOfAIFeats, int numOfACFeats, int numOfGlobalFeatures,
                                                                              IndexMap indexMap, String[] labelMap) throws Exception {
        HashMap<Object, Integer>[] rerankerFeatureVector = new HashMap[numOfAIFeats + numOfACFeats + numOfGlobalFeatures];
        for (int wordIdx = 0; wordIdx < sentence.getWords().length; wordIdx++) {
            //for each word in the sentence
            int aiLabel = (goldMap.containsKey(wordIdx)) ? 1 : 0;
            int acLabel = (goldMap.containsKey(wordIdx)) ? goldMap.get(wordIdx) : -1;
            Object[] aiFeats = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfAIFeats, indexMap, true, aiLabel);
            Object[] acFeats = acLabel == -1 ? null : FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfACFeats, indexMap, true, acLabel);
            //todo check if it works correctly
            addToRerankerFeats(rerankerFeatureVector, aiFeats, 0);
            addToRerankerFeats(rerankerFeatureVector, acFeats, numOfAIFeats);
        }
        String pLabel = sentence.getPredicatesInfo().get(pIdx);
        ArrayList<Integer> aiAssignment = new ArrayList<Integer>();
        ArrayList<Integer> acAssignment = new ArrayList<Integer>();
        for (int arg : goldMap.keySet()) {
            aiAssignment.add(arg);
            acAssignment.add(goldMap.get(arg));
        }
        Object[] globalFeats = FeatureExtractor.extractGlobalFeatures(pIdx, pLabel, new Pair<Double, ArrayList<Integer>>(1.0D, aiAssignment),
                new Pair<Double, ArrayList<Integer>>(1.0D, acAssignment), labelMap);
        addToRerankerFeats(rerankerFeatureVector, globalFeats, numOfAIFeats + numOfACFeats);
        return rerankerFeatureVector;
    }

    private class InstanceGenerator implements Runnable {
        int devPartIdx;
        ArrayList<String>[] trainParts;
        ClusterMap clusterMap;

        public InstanceGenerator(ArrayList<String>[] trainParts, ClusterMap clusterMap, int devPartIdx) {
            this.devPartIdx = devPartIdx;
            this.trainParts = trainParts;
            this.clusterMap = clusterMap;
        }

        @Override
        public void run() {
            try {
                writeTrainSentences(trainParts, clusterMap, devPartIdx);
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println(e.getMessage());
                System.exit(1);
            }
        }
    }

}