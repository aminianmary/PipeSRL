package SupervisedSRL.PD;

import SentenceStruct.PA;
import SentenceStruct.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import ml.AveragedPerceptron;

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
    public static final int maxNumOfPDIterations4UnseenPredicates = 10;

    public static void train(List<String> trainSentencesInCONLLFormat, List<String> devSentencesInCONLLFormat,
                             IndexMap indexMap, int maxNumberOfTrainingIterations, String modelDir, int numOfPDFeaturs)
            throws Exception {
        //creates lexicon of all predicates in the trainJoint set
        HashMap<Integer, HashMap<String, HashSet<Object[]>>> trainPLexicon =
                buildPredicateLexicon(trainSentencesInCONLLFormat, indexMap, numOfPDFeaturs);
        HashMap<Integer, HashMap<String, HashSet<Object[]>>> devPLexicon =
                buildPredicateLexicon(devSentencesInCONLLFormat, indexMap, numOfPDFeaturs);
        int totalNumOfPredicatesSeenInDev = 0;
        System.out.println("Training Started...");

        for (int plem : trainPLexicon.keySet()) {
            HashSet<String> possibleLabels = new HashSet<>(trainPLexicon.get(plem).keySet());
            AveragedPerceptron ap = new AveragedPerceptron(possibleLabels, numOfPDFeaturs);
            double bestAcc = 0;
            int noImprovement = 0;
            int lastIter =0;
            boolean seenInDev = false;

            for (int i = 0; i < maxNumberOfTrainingIterations; i++) {

                for (String label: trainPLexicon.get(plem).keySet()) {
                    for (Object[] instance: trainPLexicon.get(plem).get(label))
                        ap.learnInstance(instance, label);
                }

                AveragedPerceptron decodeAp = ap.calculateAvgWeights();
                //making prediction on dev instances of this plem
                int correct =0;
                int total =0;
                if (devPLexicon.containsKey(plem)){
                    //seen in dev data
                    seenInDev = true;
                    for (String goldLabel: devPLexicon.get(plem).keySet()) {
                        for (Object[] instance: devPLexicon.get(plem).get(goldLabel)){
                            String prediction = decodeAp.predict(instance);
                            total++;
                            if (prediction.equals(goldLabel))
                                correct++;
                        }
                    }
                    double acc = (double) correct/total;
                    if (acc > bestAcc) {
                        noImprovement = 0;
                        bestAcc = acc;
                        ap.saveModel(modelDir + "/" + plem);
                        lastIter = i;
                    } else {
                        noImprovement++;
                        if (noImprovement > 5) {
                            lastIter = i;
                            break;
                        }
                    }
                }else{
                    lastIter = i;
                    if (i >= maxNumOfPDIterations4UnseenPredicates)
                        break;
                }
            }
            String seenInDevStr = "unseen in dev";
            if (seenInDev==true) {
                seenInDevStr = "seen in dev";
                totalNumOfPredicatesSeenInDev++;
            }
           System.out.println ("training for plem: "+ plem +"-"+seenInDevStr +"-last iter: "+ lastIter + " acc: "+ bestAcc);
        }
        System.out.println("Total Number of Predicates seen in Dev/total Number of predicates in train data: " + totalNumOfPredicatesSeenInDev +"/" + trainPLexicon.size());
        System.out.println("Done!");
    }


    public static HashMap<Integer, String> predict(Sentence sentence, IndexMap indexMap, String modelDir, int numOfPDFeatures) throws Exception {
        //prediction assumes predicates are given (no pred ID, just pred Disambig)
        File f1;
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceLemmas = sentence.getLemmas();
        String[] sentenceLemmas_str = sentence.getLemmas_str();

        HashMap<Integer, String> predictions = new HashMap<Integer, String>();
        //given gold predicate ids, we just disambiguate them
        for (PA pa : pas) {
            totalPreds++;
            int pIdx = pa.getPredicateIndex();
            int plem = sentenceLemmas[pIdx];
            //we use coarse POS tags instead of original POS tags
            Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, sentence, numOfPDFeatures, indexMap);
            f1 = new File(modelDir + "/" + plem );
            if (f1.exists() && !f1.isDirectory()) {
                //seen predicates
                AveragedPerceptron classifier = AveragedPerceptron.loadModel(modelDir + "/" + plem );
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


    public static HashMap<Integer, HashMap<String, HashSet<Object[]>>> buildPredicateLexicon
            (List<String> sentencesInCONLLFormat, IndexMap indexMap, int numOfPDFeatures) throws Exception {
        HashMap<Integer, HashMap<String, HashSet<Object[]>>> pLexicon = new HashMap<>();

        for (int senID = 0; senID < sentencesInCONLLFormat.size(); senID++) {
            Sentence sentence = new Sentence(sentencesInCONLLFormat.get(senID), indexMap);

            ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            int[] sentenceLemmas = sentence.getLemmas();

            for (PA pa : pas) {
                int pIdx = pa.getPredicateIndex();
                int plem = sentenceLemmas[pIdx];
                String plabel = pa.getPredicateLabel();
                Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, sentence, numOfPDFeatures, indexMap);

                if (!pLexicon.containsKey(plem)) {
                    HashSet<Object[]> fvs = new HashSet<>();
                    fvs.add(pdfeats);
                    HashMap<String, HashSet<Object[]>> featureVectors = new HashMap<>();
                    featureVectors.put(plabel, fvs);
                    pLexicon.put(plem, featureVectors);

                } else{
                    if (!pLexicon.get(plem).containsKey(plabel)){
                        HashSet<Object[]> fvs = new HashSet<>();
                        fvs.add(pdfeats);
                        pLexicon.get(plem).put(plabel, fvs);
                    }else{
                        pLexicon.get(plem).get(plabel).add(pdfeats);
                    }
                }
            }
        }
        return pLexicon;
    }

}
