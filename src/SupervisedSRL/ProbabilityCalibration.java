package SupervisedSRL;
import SentenceStruct.PA;
import SentenceStruct.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Pair;
import SupervisedSRL.Strcutures.Prediction;
import com.sun.java.util.jar.pack.ConstantPool;
import ml.AveragedPerceptron;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Created by Maryam Aminian on 9/21/16.
 */
public class ProbabilityCalibration {
    int numOfBins;
    AveragedPerceptron aiClassifier;

    public ProbabilityCalibration(AveragedPerceptron classifier, int numOfBins) {
        this.numOfBins = numOfBins;
        this.aiClassifier = classifier;
    }

    private double[] obtainProbabilityEstimates4Bins (ArrayList<String> tuneSentences, IndexMap indexMap,
                                                      int aiMaxBeamSize, int numOfAIFeatures, int numOfPDFeatures,
                                                      String pdModelDir) throws Exception {
        double[] probEstimates = new double[numOfBins];
        TreeMap<Double, Boolean> sortedScores = new TreeMap<>();
        Decoder argumentDecoder = new Decoder(aiClassifier, "AI");

        for (int d = 0; d < tuneSentences.size(); d++) {
            Sentence sentence = new Sentence(tuneSentences.get(d), indexMap);
            ArrayList<PA> goldPAs = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            HashMap<Integer,HashSet<Integer>> goldPAMap = new HashMap<>();
            for (PA pa: goldPAs) {
                int pIdx = pa.getPredicateIndex();
                HashSet<Integer> args = pa.getArgumentsIndices();
                goldPAMap.put(pIdx, args);
            }

            HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, pdModelDir, numOfPDFeatures);
            int[] sentenceWords = sentence.getWords();

            for (int pIdx : predictedPredicates.keySet()) {
                for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
                    Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfAIFeatures, indexMap, false, 0);
                    double[] scores = aiClassifier.score(featVector);
                    double score0 = scores[0];
                    double score1 = scores[1];
                    if (score1 > score0){
                        //predicted as an argument
                        if (goldPAMap.get(pIdx).contains(wordIdx))
                            sortedScores.put(score1, true);
                    }else
                    {
                        //predicted as non-argument
                        sortedScores.put(score1, false);
                    }
                }
            }
        }
        //done with building the sorted map, now it's time for binning
        int binSize = (int) Math.ceil((double) sortedScores.size()/numOfBins);
        int startIndex = 0;
        int endIndex = 0;
        for (int i = 0; i < numOfBins; i++) {
            endIndex = startIndex + binSize;
            int tp;
            if (endIndex < sortedScores.size())
                partitionSentences = new ArrayList<>(trainSentences.subList(startIndex, endIndex));
            else
                partitionSentences = new ArrayList<>(trainSentences.subList(startIndex, trainSentences.size()));

            partitions[i] = partitionSentences;
            startIndex = endIndex;
        }
    }
}
