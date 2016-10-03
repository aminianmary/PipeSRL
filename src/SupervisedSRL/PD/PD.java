package SupervisedSRL.PD;

import SentenceStruct.PA;
import SentenceStruct.Predicate;
import SentenceStruct.Sentence;
import SupervisedSRL.Evaluation;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import ml.AveragedPerceptron;
import util.IO;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ExecutorService;

/**
 * Created by Maryam Aminian on 5/19/16.
 * Predicate disambiguation modules
 */
public class PD {

    public static int unseenPreds = 0;
    public static int totalPreds = 0;
    public static final int maxNumOfPDIterations4UnseenPredicates = 10;

    public static void train(ArrayList<String> trainSentencesInCONLLFormat, ArrayList<String> devSentencesInCONLLFormat,
                             IndexMap indexMap, int maxNumberOfTrainingIterations, String modelDir, int numOfPDFeaturs)
            throws Exception {
        HashMap<Integer, HashMap<String, HashSet<Object[]>>> trainPLexicon =
                buildPredicateLexicon(trainSentencesInCONLLFormat, indexMap, numOfPDFeaturs);
        HashMap<Integer, HashMap<String, HashSet<Object[]>>> devPLexicon =
                buildPredicateLexicon(devSentencesInCONLLFormat, indexMap, numOfPDFeaturs);

        System.out.println("Training Started...");

        for (int plem : trainPLexicon.keySet()) {
            HashSet<String> possibleLabels = new HashSet<>(trainPLexicon.get(plem).keySet());
            AveragedPerceptron ap = new AveragedPerceptron(possibleLabels, numOfPDFeaturs);
            double bestAcc = 0;
            int noImprovement = 0;

            for (int i = 0; i < maxNumberOfTrainingIterations; i++) {

                for (String label: trainPLexicon.get(plem).keySet()) {
                    for (Object[] instance: trainPLexicon.get(plem).get(label))
                        ap.learnInstance(instance, label);
                }

                //making prediction on dev instances of this plem
                AveragedPerceptron decodeAp = ap.calculateAvgWeights();
                int correct =0;
                int total =0;
                if (devPLexicon.containsKey(plem)){
                    //seen in dev data
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
                    } else {
                        noImprovement++;
                        if (noImprovement > 5) {
                            break;
                        }
                    }
                }else{
                    if (i >= maxNumOfPDIterations4UnseenPredicates)
                        break;
                }
            }
        }
        System.out.println("Done!");
        Evaluation.evaluatePD(devSentencesInCONLLFormat, modelDir, indexMap, numOfPDFeaturs);
    }


    public static void predict (ArrayList<String> sentencesInCONLLFormat, IndexMap indexMap,
                                                      String modelDir, int numOfPDFeatures, String path2SavePredictions)
            throws Exception{
        HashMap<Integer, String>[] pdPredictions = new HashMap[sentencesInCONLLFormat.size()];

        for (int d=0; d< sentencesInCONLLFormat.size(); d++){
            Sentence sentence = new Sentence(sentencesInCONLLFormat.get(d), indexMap);
            pdPredictions[d] =predict4ThisSentence(sentence, indexMap, modelDir, numOfPDFeatures);
        }
        IO.write(pdPredictions, path2SavePredictions);
    }


    public static HashMap<Integer, String> predict4ThisSentence(Sentence sentence, IndexMap indexMap, String modelDir,
                                                                int numOfPDFeatures) throws Exception {
        //prediction assumes predicates are given (no pred ID, just pred Disambig)
        File f1;
        ArrayList<Predicate> predicates = sentence.getPredicates();
        int[] sentenceLemmas = sentence.getLemmas();
        String[] sentenceLemmas_str = sentence.getLemmas_str();

        HashMap<Integer, String> predictions = new HashMap<Integer, String>();
        for (Predicate p: predicates) {
            totalPreds++;
            int pIdx = p.getIndex();
            assert p.getPredicateGoldLabel() == null;
            int plem = sentenceLemmas[pIdx];
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
            ArrayList<Predicate> predicates = sentence.getPredicates();
            int[] sentenceLemmas = sentence.getLemmas();

            for (Predicate p: predicates) {
                int pIdx = p.getIndex();
                int plem = sentenceLemmas[pIdx];
                //todo this should be the only place that predicate gold label is used
                String pGoldLabel = p.getPredicateGoldLabel();
                assert pGoldLabel!= null;
                Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, sentence, numOfPDFeatures, indexMap);

                if (!pLexicon.containsKey(plem)) {
                    HashSet<Object[]> fvs = new HashSet<>();
                    fvs.add(pdfeats);
                    HashMap<String, HashSet<Object[]>> featureVectors = new HashMap<>();
                    featureVectors.put(pGoldLabel, fvs);
                    pLexicon.put(plem, featureVectors);

                } else{
                    if (!pLexicon.get(plem).containsKey(pGoldLabel)){
                        HashSet<Object[]> fvs = new HashSet<>();
                        fvs.add(pdfeats);
                        pLexicon.get(plem).put(pGoldLabel, fvs);
                    }else{
                        pLexicon.get(plem).get(pGoldLabel).add(pdfeats);
                    }
                }
            }
        }
        return pLexicon;
    }

}
