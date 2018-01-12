package SupervisedSRL;

import SentenceStruct.Sentence;
import SentenceStruct.simplePA;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.*;
import ml.AveragedPerceptron;
import util.IO;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by Maryam Aminian on 5/24/16.
 */
public class PDecoder {

    AveragedPerceptron piClassifier;
    public PDecoder(AveragedPerceptron piClassifier) {
        this.piClassifier = piClassifier;
    }

    ////////////////////////////////// DECODE ////////////////////////////////////////////////////////

    public void decode(IndexMap indexMap, ArrayList<String> devSentencesInCONLLFormat,
                       int aiMaxBeamSize, int acMaxBeamSize, int numOfPIFeatures, int numOfPDFeatures,
                       int numOfAIFeatures, int numOfACFeatures, String outputFile,String outputFileWithSourceInfo,
                       double aiCoefficient,
                       String pdModelDir, boolean usePI, boolean supplement) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");
        System.out.println("Decoding started (on dev data)...");
        long startTime = System.currentTimeMillis();
        BufferedWriter outputWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "UTF-8"));
        BufferedWriter outputWithProjectedInfoWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFileWithSourceInfo), "UTF-8"));
        BufferedWriter outputScoresWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile+".score"), "UTF-8"));

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());

            String devSentence = devSentencesInCONLLFormat.get(d);
            Sentence sentence = new Sentence(devSentence, indexMap);

            TreeMap<Integer, simplePA> prediction = (TreeMap<Integer, simplePA>) predict(sentence, indexMap,
                    aiMaxBeamSize, acMaxBeamSize, numOfPIFeatures, numOfPDFeatures, numOfAIFeatures,
                    numOfACFeatures, false, aiCoefficient, pdModelDir, usePI);

            SRLOutput output = IO.generateCompleteOutputSentenceInCoNLLFormat(sentence, devSentence,prediction,supplement);
            outputWriter.write(output.getSentence());
            outputWithProjectedInfoWriter.write(output.getSentence_w_projected_info());
            outputScoresWriter.write(d+"\t"+ output.getConfidenceScore() +"\n");
        }
        System.out.println(devSentencesInCONLLFormat.size());
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
        outputWriter.flush();
        outputWriter.close();
        outputWithProjectedInfoWriter.flush();
        outputWithProjectedInfoWriter.close();
        outputScoresWriter.flush();
        outputScoresWriter.close();
    }

    public Object predict(Sentence sentence, IndexMap indexMap, int numOfPIFeatures, int numOfPDFeatures,
                          String pdModelDir, boolean usePI) throws Exception {

        TreeMap<Integer, simplePA> predictedPAs = new TreeMap<Integer, simplePA>();
        int[] sentenceLemmas = sentence.getLemmas();
        String[] sentenceLemmas_str = sentence.getLemmas_str();
        ArrayList<Integer> goldPredicateIndices = sentence.getPredicatesIndices(); // no predicate ID

        for (int wordIdx = 1; wordIdx < sentence.getLength(); wordIdx++) {
            boolean isPredicate = false;
            if (usePI) {
                Object[] featureVector = FeatureExtractor.extractPIFeatures(wordIdx, sentence, numOfPIFeatures, indexMap);
                String piPrediction = piClassifier.predict(featureVector);
                if (piPrediction.equals("1"))
                    isPredicate = true;
            }else{
                if(goldPredicateIndices.contains(wordIdx))
                    isPredicate = true;
            }

            if (isPredicate) {
                //identified as a predicate
                int pIdx = wordIdx;
                int plem = sentenceLemmas[pIdx];
                String pLabel = "";

                Object[] pdfeats = FeatureExtractor.extractPDFeatures(pIdx, sentence, numOfPDFeatures, indexMap);
                File f1 = new File(pdModelDir + "/" + plem);
                if (f1.exists() && !f1.isDirectory()) {
                    //seen predicates
                    AveragedPerceptron classifier = AveragedPerceptron.loadModel(pdModelDir + "/" + plem);
                    pLabel = classifier.predict(pdfeats);
                } else {
                    if (plem != indexMap.unknownIdx) {
                        pLabel = indexMap.int2str(plem) + ".01"; //seen pLem
                    } else {
                        pLabel = sentenceLemmas_str[pIdx] + ".01"; //unseen pLem
                    }
                }

                //having pd label, set pSense in the sentence
                sentence.setPDAutoLabels4ThisPredicate(pIdx, pLabel);
                ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfAIFeatures);
                ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = getBestACCandidates(sentence,
                        pIdx, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures, aiCoefficient);

                if (use4Reranker)
                    predictedAIACCandidates.put(pIdx, new Prediction4Reranker(pLabel, aiCandidates, acCandidates));
                else {
                    HashMap<Integer, String> highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates, labelMap);
                    predictedPAs.put(pIdx, new simplePA(pLabel, highestScorePrediction));
                }
            }
        }

        if (use4Reranker)
            return predictedAIACCandidates;
        else
            return predictedPAs;
    }
}