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
                             IndexMap indexMap, int maxNumberOfTrainingIterations, String modelDir, int numOfPDFeatures)
            throws Exception {
        HashMap<Integer, HashMap<String, HashSet<Object[]>>> trainPLexicon =
                buildPredicateLexicon(trainSentencesInCONLLFormat, indexMap, numOfPDFeatures);
        HashMap<Integer, HashMap<String, HashSet<Object[]>>> devPLexicon =
                buildPredicateLexicon(devSentencesInCONLLFormat, indexMap, numOfPDFeatures);
        int numOfSavedModelFiles =0;
        System.out.println("Training Started...");
        System.out.println("trainPLexicon num of lemmas: " + trainPLexicon.size());

        for (int plem : trainPLexicon.keySet()) {
            HashSet<String> possibleLabels = new HashSet<>(trainPLexicon.get(plem).keySet());
            AveragedPerceptron ap = new AveragedPerceptron(possibleLabels, numOfPDFeatures);
            double completeness =1;
            double bestAcc = 0;
            int noImprovement = 0;
            boolean savedModel4ThisLemma = false;

            for (int i = 0; i < maxNumberOfTrainingIterations; i++) {

                for (String label: trainPLexicon.get(plem).keySet()) {
                    for (Object[] instance : trainPLexicon.get(plem).get(label)) {
                        ap.learnInstance(instance, label, completeness);
                    }
                }
                //making prediction on dev instances of this plem
                if (devPLexicon.containsKey(plem)){
                    //seen in dev data
                    AveragedPerceptron decodeAp = ap.calculateAvgWeights();
                    int correct =0;
                    int total =0;

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
                        savedModel4ThisLemma = true;
                    } else {
                        if (bestAcc == 0) {
                            ap.saveModel(modelDir + "/" + plem);
                            savedModel4ThisLemma = true;
                        }
                        else {
                            noImprovement++;
                            if (noImprovement > 5) {
                                break;
                            }
                        }
                    }
                }else{
                    if (i >= maxNumOfPDIterations4UnseenPredicates) {
                        ap.saveModel(modelDir + "/" + plem);
                        savedModel4ThisLemma = true;
                        break;
                    }
                }
            }
            if (savedModel4ThisLemma == true)
                numOfSavedModelFiles++;
        }
        System.out.println("Number of saved models for lemmas in the train data " + numOfSavedModelFiles);
        System.out.println("Done!");
    }

    public static void predict (ArrayList<String> sentencesInCONLLFormat, IndexMap indexMap,
                                                      String modelDir, int numOfPDFeatures, String path2SavePredictions)
            throws Exception{
        HashMap<Integer, String>[] pdPredictions = new HashMap[sentencesInCONLLFormat.size()];
        int total =0;
        int correct=0;

        for (int d=0; d< sentencesInCONLLFormat.size(); d++){
            if (d%1000 ==0)
                System.out.print(d+"...");

            Sentence sentence = new Sentence(sentencesInCONLLFormat.get(d), indexMap);
            HashMap<Integer, String> goldPredicateLabelMap = sentence.getPredicatesGoldLabelMap();
            pdPredictions[d] =predict4ThisSentence(sentence, indexMap, modelDir, numOfPDFeatures);
            assert goldPredicateLabelMap.size() == pdPredictions[d].size();
            total += goldPredicateLabelMap.size();

            for (int pIdx: goldPredicateLabelMap.keySet()) {
                assert pdPredictions[d].containsKey(pIdx);
                if (goldPredicateLabelMap.get(pIdx).equals(pdPredictions[d].get(pIdx)))
                    correct++;
            }
        }
        System.out.print(sentencesInCONLLFormat.size()+"\n");
        double acc = ((double) correct/total) *100;
        System.out.print("PD Accuracy: " + acc);
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
                if (plem != indexMap.unknownIdx) {
                    predictions.put(pIdx, indexMap.int2str(plem) + ".01"); //seen pLem
                }
                else {
                    predictions.put(pIdx, sentenceLemmas_str[pIdx] + ".01"); //unseen pLem
                }
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
                if (pIdx == -1)
                    System.out.print("****NOTE!*** Predicate Index is -1 in sentence " + senID+ "\n");
                int plem = sentenceLemmas[pIdx];
                String pGoldLabel = p.getPredicateGoldLabel();
                assert pGoldLabel != null;
                Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, sentence, numOfPDFeatures, indexMap);

                if (!pLexicon.containsKey(plem)) {
                    HashSet<Object[]> fvs = new HashSet<>();
                    fvs.add(pdfeats);
                    HashMap<String, HashSet<Object[]>> featureVectors = new HashMap<>();
                    featureVectors.put(pGoldLabel, fvs);
                    pLexicon.put(plem, featureVectors);

                } else {
                    if (!pLexicon.get(plem).containsKey(pGoldLabel)) {
                        HashSet<Object[]> fvs = new HashSet<>();
                        fvs.add(pdfeats);
                        pLexicon.get(plem).put(pGoldLabel, fvs);
                    } else {
                        pLexicon.get(plem).get(pGoldLabel).add(pdfeats);
                    }
                }
            }
        }
        return pLexicon;
    }

}
