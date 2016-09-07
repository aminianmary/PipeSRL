package SupervisedSRL.PD;

import SentStructs.PA;
import SentStructs.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Pipeline;
import SupervisedSRL.Strcutures.ClusterMap;
import SupervisedSRL.Strcutures.IndexMap;
import ml.AveragedPerceptron;
import util.IO;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 * Created by Maryam Aminian on 5/19/16.
 * Predicate disambiguation modules
 */
public class PD {

    public static int unseenPreds = 0;
    public static int totalPreds = 0;

    public static void main(String[] args) throws Exception {

        String inputFile = args[0];
        String modelDir = args[1];
        String clusterFile = args[2];

        int numOfPDFeatures = 9;

        ArrayList<String> sentencesInCONLLFormat = IO.readCoNLLFile(inputFile);
        final ClusterMap clusterMap = new ClusterMap(clusterFile);
        final IndexMap indexMap = new IndexMap(sentencesInCONLLFormat, clusterMap, Pipeline.numOfACFeatures, false);

        int totalNumOfSentences = sentencesInCONLLFormat.size();
        int trainSize = (int) Math.floor(0.8 * totalNumOfSentences);

        List<String> train = sentencesInCONLLFormat.subList(0, trainSize);
        List<String> test = sentencesInCONLLFormat.subList(trainSize, totalNumOfSentences);

        //training
        train(train, indexMap, clusterMap, 10, modelDir, numOfPDFeatures);

        //prediction
        HashMap<Integer, String>[] predictions = new HashMap[test.size()];
        System.out.println("Prediction started...");
        for (int senIdx = 0; senIdx < test.size(); senIdx++) {
            boolean decode = true;
            Sentence sentence = new Sentence(test.get(senIdx), indexMap, clusterMap);
            predictions[senIdx] = predict(sentence, indexMap, modelDir, numOfPDFeatures);
        }

    }


    public static void train(List<String> trainSentencesInCONLLFormat, IndexMap indexMap, ClusterMap clusterMap, int numberOfTrainingIterations, String modelDir, int numOfPDFeaturs)
            throws Exception {
        //creates lexicon of all predicates in the trainJoint set
        HashMap<Integer, HashMap<Integer, HashSet<PredicateLexiconEntry>>> trainPLexicon =
                buildPredicateLexicon(trainSentencesInCONLLFormat, indexMap, clusterMap, numOfPDFeaturs);

        System.out.println("Training Started...");

        for (int plem : trainPLexicon.keySet()) {
            //extracting feature vector for each training example
            for (int ppos : trainPLexicon.get(plem).keySet()) {
                HashSet<PredicateLexiconEntry> featVectors = trainPLexicon.get(plem).get(ppos);
                HashSet<String> labelSet = getLabels(featVectors);

                AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfPDFeaturs);

                //System.out.print("training model for predicate/pos -->"+ plem+"|"+ppos+"\n");
                for (int i = 0; i < numberOfTrainingIterations; i++) {
                    //System.out.print("iteration:" + i + "...");
                    for (PredicateLexiconEntry ple : trainPLexicon.get(plem).get(ppos)) {
                        //trainJoint average perceptron
                        String plabel = ple.getPlabel();
                        ap.learnInstance(ple.getPdfeats(), plabel);
                    }
                }
                ap.saveModel(modelDir + "/" + plem + "_" + ppos);
            }
        }
        System.out.println("Done!");
    }

    public static HashMap<Integer, String> predict(Sentence sentence, IndexMap indexMap, String modelDir, int numOfPDFeatures) throws Exception {
        File f1;
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceLemmas = sentence.getLemmas();
        int[] sentencePOSTags = sentence.getPosTags();
        int[] sentenceCPOSTags = sentence.getCPosTags();
        String[] sentenceLemmas_str = sentence.getLemmas_str();

        HashMap<Integer, String> predictions = new HashMap<Integer, String>();
        //given gold predicate ids, we just disambiguate them
        for (PA pa : pas) {
            totalPreds++;
            int pIdx = pa.getPredicateIndex();
            int plem = sentenceLemmas[pIdx];
            //we use coarse POS tags instead of original POS tags
            int ppos = sentenceCPOSTags[pIdx]; //sentencePOSTags[pIdx];
            Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, sentence, numOfPDFeatures, indexMap);
            f1 = new File(modelDir + "/" + plem + "_" + ppos);
            if (f1.exists() && !f1.isDirectory()) {
                //seen predicates
                AveragedPerceptron classifier = AveragedPerceptron.loadModel(modelDir + "/" + plem + "_" + ppos);
                String prediction = classifier.predict(pdfeats);
                predictions.put(pIdx, prediction);
            } else {
                //unseen predicate --> assign lemma.01 (default sense) as predicate label instead of null
                unseenPreds++;
                if (plem != indexMap.unknownIdx)
                    predictions.put(pIdx, indexMap.int2str(plem) + ".01"); //seen pLem
                else
                    predictions.put(pIdx, sentenceLemmas_str[pIdx] + ".01"); //unseen pLem
            }
        }
        return predictions;
    }


    public static HashMap<Integer, HashMap<Integer, HashSet<PredicateLexiconEntry>>> buildPredicateLexicon
            (List<String> sentencesInCONLLFormat, IndexMap indexMap, ClusterMap clusterMap, int numOfPDFeatures) throws Exception {
        HashMap<Integer, HashMap<Integer, HashSet<PredicateLexiconEntry>>> pLexicon = new HashMap<Integer, HashMap<Integer, HashSet<PredicateLexiconEntry>>>();

        boolean decode = false;
        for (int senID = 0; senID < sentencesInCONLLFormat.size(); senID++) {
            Sentence sentence = new Sentence(sentencesInCONLLFormat.get(senID), indexMap, clusterMap);

            ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            int[] sentenceLemmas = sentence.getLemmas();
            int[] sentenceCPOSTags = sentence.getCPosTags();

            for (PA pa : pas) {
                int pIdx = pa.getPredicateIndex();
                int plem = sentenceLemmas[pIdx];
                String plabel = pa.getPredicateLabel();
                //instead of original POS tags, we use coarse POS tags
                int ppos = sentenceCPOSTags[pIdx];

                Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, sentence, numOfPDFeatures, indexMap);
                PredicateLexiconEntry ple = new PredicateLexiconEntry(plabel, pdfeats);

                if (!pLexicon.containsKey(plem)) {
                    HashMap<Integer, HashSet<PredicateLexiconEntry>> posDic = new HashMap<Integer, HashSet<PredicateLexiconEntry>>();
                    HashSet<PredicateLexiconEntry> featVectors = new HashSet<PredicateLexiconEntry>();
                    featVectors.add(ple);
                    posDic.put(ppos, featVectors);
                    pLexicon.put(plem, posDic);

                } else if (!pLexicon.get(plem).containsKey(ppos)) {
                    HashSet<PredicateLexiconEntry> featVectors = new HashSet<PredicateLexiconEntry>();
                    featVectors.add(ple);
                    pLexicon.get(plem).put(ppos, featVectors);
                } else {
                    pLexicon.get(plem).get(ppos).add(ple);
                }

            }
        }

        return pLexicon;
    }


    public static HashSet<String> getLabels(HashSet<PredicateLexiconEntry> featVectors) {
        HashSet<String> labelSet = new HashSet<String>();
        for (PredicateLexiconEntry ple : featVectors)
            labelSet.add(ple.getPlabel());

        return labelSet;
    }

}
