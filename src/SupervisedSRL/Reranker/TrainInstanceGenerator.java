package SupervisedSRL.Reranker;
import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.GZIPOutputStream;

import SupervisedSRL.Features.FeatureExtractor;
import Sentence.Sentence;
import SupervisedSRL.Decoder;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.*;
import SupervisedSRL.Train;
import ml.AveragedPerceptron;
import Sentence.*;
import util.IO;

/**
 * Created by Maryam Aminian on 8/19/16.
 */
public class TrainInstanceGenerator {
    ArrayList<ArrayList<String>> trainPartitions;
    int numOfPartitions;
    String clusterFile;
    String modelDir;
    int numOfPDFeatures;
    int numOfPDTrainingIterations;
    int numberOfTrainingIterations;
    int numOfAIFeatures;
    int numOfACFeatures;
    int numOfGlobalFeatures;
    int aiMaxBeamSize;
    int acMaxBeamSize;
    boolean greedy;

    public static void main(String[] args) throws Exception {
        String trainFilePath = args[0];
        String clusterFilePath = args[1];
        String modelDir = args[2];
        int numOfTrainingIterations = Integer.parseInt(args[3]);
        int aiBeamSize = Integer.parseInt(args[4]);
        int acBeamSize = Integer.parseInt(args[5]);
        boolean greedy= Boolean.parseBoolean(args[6]);
        int numOfPartitions = Integer.parseInt(args[7]);

        int numOfPDIterations =10;
        int numOfPDFeatures = 9;
        int numOfAIFeatures = 25 + 3+ 5+ 13;
        int numOfACFeatures = 25 + 3+ 5+ 15;
        int numOfGlobalFeatures= 1;

        TrainInstanceGenerator trainInstanceGenerator = new TrainInstanceGenerator(trainFilePath, numOfPartitions,
                clusterFilePath, modelDir, numOfPDFeatures, numOfPDIterations, numOfTrainingIterations, numOfAIFeatures,
                numOfACFeatures, numOfGlobalFeatures, aiBeamSize, acBeamSize,greedy);
        trainInstanceGenerator.buildTrainInstances();
    }

    public TrainInstanceGenerator(String trainFilePath, int numOfParts, String clusterFile, String modelDir,
            int numOfPDFeatures, int numOfPDTrainingIterations, int numberOfTrainingIterations,
            int numOfAIFeatures, int numOfACFeatures, int numOfGlobalFeatures, int aiMaxBeamSize,
            int acMaxBeamSize, boolean greedy) throws IOException {

        this.numOfPartitions =  numOfParts;
        this.clusterFile = clusterFile;
        this. modelDir= modelDir;
        this.numOfPDFeatures = numOfPDFeatures;
        this. numOfPDTrainingIterations = numOfPDTrainingIterations;
        this.numberOfTrainingIterations= numberOfTrainingIterations;
        this.numOfAIFeatures= numOfAIFeatures;
        this.numOfACFeatures= numOfACFeatures;
        this.numOfGlobalFeatures= numOfGlobalFeatures;
        this.aiMaxBeamSize= aiMaxBeamSize;
        this.acMaxBeamSize= acMaxBeamSize;
        this.greedy= greedy;
        ArrayList<ArrayList<String>> partitions = getPartitions(trainFilePath);
        this.trainPartitions = partitions;
    }


    private ArrayList<ArrayList<String>> getPartitions(String trainFilePath) throws IOException {
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
        return partitions;
    }


    public void buildTrainInstances () throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(numOfPartitions);
        CompletionService<Boolean> pool =  new ExecutorCompletionService<Boolean>(executor);

        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            pool.submit(new InstanceGenerator(devPartIdx));
        }
        System.out.println("Reranker training instance generation are submitted");
        for (int devPartIdx = 0; devPartIdx < numOfPartitions; devPartIdx++) {
            assert pool.take().get();
            System.out.println((devPartIdx+1)+" jobs done!");
        }
        System.out.println("All jobs done!");
    }


    private void writeTrainSentences(int devPartIdx) throws Exception {
        ArrayList<String> trainSentences = new ArrayList<String>();
        ArrayList<String> devSentences = new ArrayList<String>();

        for (int partIdx = 0; partIdx < numOfPartitions; partIdx++) {
            if (partIdx == devPartIdx)
                devSentences = trainPartitions.get(partIdx);
            else
                trainSentences.addAll(trainPartitions.get(partIdx));
        }
        //write dev sentences into a file (to be compatible with previous functions)
        String devDataPath = modelDir + "/reranker_dev_" + devPartIdx;
        BufferedWriter devWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(devDataPath)));
        for (String sentence : devSentences)
            devWriter.write(sentence + "\n\n");
        devWriter.flush();
        devWriter.close();

        //train a PD-AI-AC modules on the train parts
        HashSet<String> argLabels = IO.obtainLabels(trainSentences);
        final IndexMap indexMap = new IndexMap(trainSentences, clusterFile);
        PD.train(trainSentences, indexMap, numOfPDTrainingIterations, modelDir, numOfPDFeatures);
        String aiModelPath = Train.trainAI(trainSentences, devSentences, indexMap,
                numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfPDFeatures, aiMaxBeamSize, greedy);
        String acModelPath = Train.trainAC(trainSentences, devDataPath, argLabels, indexMap,
                numberOfTrainingIterations, modelDir, numOfAIFeatures, numOfACFeatures, numOfPDFeatures,
                aiMaxBeamSize, acMaxBeamSize, greedy);

        //decode on dev part
        ModelInfo aiModelInfo = new ModelInfo(aiModelPath);
        AveragedPerceptron aiClassifier = aiModelInfo.getClassifier();
        AveragedPerceptron acClassifier = AveragedPerceptron.loadModel(acModelPath);
        Decoder decoder = new Decoder(aiClassifier, acClassifier);
        ArrayList<RerankerPool> rerankerPoolsInThisDevPart = new ArrayList<RerankerPool>();

        System.out.println("Decoding started on " + devDataPath + "...\n");

        for (int d = 0; d < devSentences.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentences.size());

            String devSentence = devSentences.get(d);
            Sentence sentence = new Sentence(devSentence, indexMap);
            HashMap<Integer, HashMap<Integer, Integer>> goldMap = getGoldArgLabelMap(sentence, acClassifier.getReverseLabelMap());

            TreeMap<Integer, Prediction4Reranker> predictedAIACCandidates4thisSen =
                    (TreeMap<Integer, Prediction4Reranker>) decoder.predict(sentence, indexMap, aiMaxBeamSize, acMaxBeamSize,
                            numOfAIFeatures, numOfACFeatures, numOfPDFeatures, modelDir,
                            null, null, ClassifierType.AveragedPerceptron, greedy, true);

            //creating the pool
            for (int pIdx : predictedAIACCandidates4thisSen.keySet()) {
                RerankerPool rerankerPool = new RerankerPool();
                String pLabel = predictedAIACCandidates4thisSen.get(pIdx).getPredicateLabel();
                //to keep scores/feats of different assignments
                HashMap<Integer, Integer> goldMap4ThisPredicate = goldMap.get(pIdx);

                ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = predictedAIACCandidates4thisSen.get(pIdx).getAiCandidates();
                ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = predictedAIACCandidates4thisSen.get(pIdx).getAcCandidates();

                for (int i = 0; i < aiCandidates.size(); i++) {
                    Pair<Double, ArrayList<Integer>> aiCandid = aiCandidates.get(i);
                    ArrayList<Pair<Double, ArrayList<Integer>>> acCandids4thisAiCandid = acCandidates.get(i);

                    for (int j = 0; j < acCandids4thisAiCandid.size(); j++) {
                        Pair<Double, ArrayList<Integer>> acCandid = acCandids4thisAiCandid.get(j);
                        rerankerPool.addInstance(new RerankerInstanceItem(extractRerankerFeatures(pIdx, pLabel, sentence, aiCandid, acCandid,
                                numOfAIFeatures, numOfACFeatures, indexMap, acClassifier.getLabelMap()), "0"), false);
                    }
                }
                //add gold assignment to the pool
                rerankerPool.addInstance(new RerankerInstanceItem(extractRerankerFeatures4GoldAssignment(pIdx, sentence, goldMap4ThisPredicate,
                        numOfAIFeatures, numOfACFeatures, numOfGlobalFeatures, indexMap, acClassifier.getLabelMap()), "1"), true);
                rerankerPoolsInThisDevPart.add(rerankerPool);
            }
        }
        //write dev pools
        System.out.println("Writing rerankerPools for dev part "+ devPartIdx+"\n");
        String rerankerPoolsFilePath = modelDir+ "/rerankerPools_"+ devPartIdx;
        writeRerankerPools(rerankerPoolsInThisDevPart, rerankerPoolsFilePath);
    }


    private void writeRerankerPools (ArrayList<RerankerPool> rerankerPools,String filePath) throws IOException{
        DecimalFormat format = new DecimalFormat("##.00");
        FileOutputStream fos = new FileOutputStream(filePath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);
        long startTime = System.currentTimeMillis();
        writer.writeObject(rerankerPools);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time to save pools: " + format.format( ((endTime - startTime)/1000.0)/ 60.0));
    }


    private HashMap<Integer, Integer> getArgLabelMap (Pair<Double, ArrayList<Integer>> aiCandid,
                                                      Pair<Double, ArrayList<Integer>> acCandid) {
        HashMap<Integer, Integer> argLabelMap = new HashMap<Integer, Integer>();
        assert aiCandid.second.size()== acCandid.second.size();
        for (int i = 0; i < aiCandid.second.size(); i++) {
            int wordIdx = aiCandid.second.get(i);
            int label = acCandid.second.get(i);
            assert !argLabelMap.containsKey(wordIdx);
            argLabelMap.put(wordIdx, label);
        }
        return argLabelMap;
    }

    private HashMap<Integer, HashMap<Integer, Integer>> getGoldArgLabelMap (Sentence sentence,
                                                                            HashMap<String, Integer> reverseLabelMap){
        HashMap<Integer, HashMap<Integer, Integer>> goldArgLabelMap = new HashMap<Integer, HashMap<Integer, Integer>>();
        ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        for (PA pa: goldPAs){
            int goldPIdx = pa.getPredicateIndex();
            ArrayList<Argument> goldArgs=  pa.getArguments();
            HashMap<Integer, Integer> goldArgMap = new HashMap<Integer, Integer>();
            for (Argument arg: goldArgs)
                goldArgMap.put(arg.getIndex(), reverseLabelMap.get(arg.getType()));
            goldArgLabelMap.put(goldPIdx, goldArgMap);
        }
        return goldArgLabelMap;
    }

    private HashMap<Object, Integer>[] extractRerankerFeatures (int pIdx, String pLabel, Sentence sentence, Pair<Double, ArrayList<Integer>> aiCandid, Pair<Double,
            ArrayList<Integer>> acCandid, int numOfAIFeats, int numOfACFeats, IndexMap indexMap, String[] labelMap) throws Exception {
        HashMap<Integer, Integer> argMap = getArgLabelMap(aiCandid, acCandid);
        int numOfGlobalFeatures=1;
        HashMap<Object, Integer>[] rerankerFeatureVector = new HashMap[numOfAIFeats+ numOfACFeats + numOfGlobalFeatures];
        for (int wordIdx =0 ; wordIdx< sentence.getWords().length; wordIdx++){
            //for each word in the sentence
            int aiLabel = (argMap.containsKey(wordIdx))? 1:0;
            int acLabel = (argMap.containsKey(wordIdx))? argMap.get(wordIdx):-1;
            Object[] aiFeats = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence,numOfAIFeats,indexMap, true,aiLabel);
            Object[] acFeats = acLabel==-1?null : FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence,numOfACFeats,indexMap, true,acLabel);
            //todo check if it works correctly
            addToRerankerFeats(rerankerFeatureVector, aiFeats, 0);
            addToRerankerFeats(rerankerFeatureVector, acFeats, aiFeats.length);
        }
        Object[] globalFeats = FeatureExtractor.extractGlobalFeatures(pIdx, pLabel,aiCandid, acCandid, labelMap);
        addToRerankerFeats(rerankerFeatureVector, globalFeats, numOfAIFeats+ numOfACFeats);
        return rerankerFeatureVector;
    }


    private HashMap<Object, Integer>[] extractRerankerFeatures4GoldAssignment (int pIdx,
                                                                              Sentence sentence, HashMap<Integer, Integer> goldMap,
                                                                              int numOfAIFeats, int numOfACFeats, int numOfGlobalFeatures,
                                                                              IndexMap indexMap, String[] labelMap) throws Exception {
        HashMap<Object, Integer>[] rerankerFeatureVector = new HashMap[numOfAIFeats+ numOfACFeats + numOfGlobalFeatures];
        for (int wordIdx =0 ; wordIdx< sentence.getWords().length; wordIdx++){
            //for each word in the sentence
            int aiLabel = (goldMap.containsKey(wordIdx))? 1:0;
            int acLabel = (goldMap.containsKey(wordIdx))? goldMap.get(wordIdx):-1;
            Object[] aiFeats = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence,numOfAIFeats,indexMap, true,aiLabel);
            Object[] acFeats = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence,numOfACFeats,indexMap, true,acLabel);
            //todo check if it works correctly
            addToRerankerFeats(rerankerFeatureVector, aiFeats, 0);
            addToRerankerFeats(rerankerFeatureVector, acFeats, numOfAIFeats);
        }
        String pLabel = sentence.getPredicatesInfo().get(pIdx);
        ArrayList<Integer> aiAssignment = new ArrayList<Integer>();
        ArrayList<Integer> acAssignment = new ArrayList<Integer>();
        for (int arg: goldMap.keySet()) {
            aiAssignment.add(arg);
            acAssignment.add(goldMap.get(arg));
        }
        Object[] globalFeats = FeatureExtractor.extractGlobalFeatures(pIdx, pLabel, new Pair<Double, ArrayList<Integer>>(1.0D, aiAssignment),
                new Pair<Double, ArrayList<Integer>>(1.0D, acAssignment), labelMap);
        addToRerankerFeats(rerankerFeatureVector, globalFeats, numOfAIFeats+ numOfACFeats);
        return rerankerFeatureVector;
    }


    private void addToRerankerFeats (HashMap<Object, Integer>[] rerankerFeatureVector, Object[] feats, int offset)
    {
        if(feats==null) return;
        for (int i=0; i< feats.length; i++){
            if (!rerankerFeatureVector[offset+i].containsKey(feats[i]))
                rerankerFeatureVector[offset+i].put(feats[i],1);
            else
                rerankerFeatureVector[offset+i].put(feats[i], rerankerFeatureVector[offset+i].get(feats[i])+1);
        }
    }

    private class InstanceGenerator implements Callable<Boolean> {
        int devPartIdx;

        public InstanceGenerator(int devPartIdx) {
            this.devPartIdx = devPartIdx;
        }

        @Override
        public Boolean call() throws Exception {
            writeTrainSentences(devPartIdx);
            return true;
        }
    }
}