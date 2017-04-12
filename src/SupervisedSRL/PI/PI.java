package SupervisedSRL.PI;
import SentenceStruct.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.ModelInfo;
import SupervisedSRL.Strcutures.ProjectConstants;
import ml.AveragedPerceptron;
import util.IO;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by Maryam Aminian on 10/21/16.
 */
public class PI {
    public static HashMap<String, Double> confusionMatrix;

    public static void train(ArrayList<String> trainSentencesInCONLLFormat, ArrayList<String> devSentencesInCONLLFormat,
                             IndexMap indexMap, int maxNumberOfTrainingIterations, String PIModelPath,
                             int numOfPIFeatures, String weightedLearning, String confusionMatrixPath) throws Exception {
        confusionMatrix = IO.loadConfusionMatrix(confusionMatrixPath);
        HashSet<String> labelSet = new HashSet<String>();
        labelSet.add("1");
        labelSet.add("0");
        AveragedPerceptron ap = new AveragedPerceptron(labelSet, numOfPIFeatures);
        double bestAcc = 0;
        int noImprovement = 0;

        for (int iter = 0; iter < maxNumberOfTrainingIterations; iter++) {
            System.out.print("iteration:" + iter + "\n");

            for (int sIdx = 0; sIdx < trainSentencesInCONLLFormat.size(); sIdx++) {
                if (sIdx % 1000 ==0)
                    System.out.print(sIdx+"...");
                Sentence sentence = new Sentence(trainSentencesInCONLLFormat.get(sIdx), indexMap);
                int[] sentenceSourceDepLabels = sentence.getSourceDepLabels();
                int[] sentenceDepLabels = sentence.getDepLabels();

                //double learningWeight = (weightedLearning)? sentence.getCompletenessDegree() :1;
                ArrayList<Integer> goldPredicateIndices = sentence.getPredicatesIndices();
                String[] sentenceFillPredicate = sentence.getFillPredicate();

                for (int wordIdx = 1; wordIdx < sentence.getLength(); wordIdx++) {
                    if (!sentenceFillPredicate[wordIdx].equals("?"))
                    {
                        String label = (goldPredicateIndices.contains(wordIdx)) ? "1" : "0";
                        Object[] featureVector = FeatureExtractor.extractPIFeatures(wordIdx, sentence, numOfPIFeatures, indexMap);
                        double learningWeight = 1;
                        if (weightedLearning.equals("dep"))
                            learningWeight = (sentenceDepLabels[wordIdx] == sentenceSourceDepLabels[wordIdx])? 1: 0.5;
                        else if (weightedLearning.equals("sparse"))
                            learningWeight = sentence.getCompletenessDegree();
                        else if (weightedLearning.equals("sdep")){
                            double depWeight = (sentenceDepLabels[wordIdx] == sentenceSourceDepLabels[wordIdx])? 1: 0.5;
                            double sparsityWeight = sentence.getCompletenessDegree();
                            learningWeight = 2 * (depWeight*sparsityWeight)/(depWeight+sparsityWeight);
                        }
                        else if (weightedLearning.equals("cm")){
                            String sourceDepLabel = indexMap.int2str(sentenceDepLabels[wordIdx]);
                            String targetDepLabel = indexMap.int2str(sentenceSourceDepLabels[wordIdx]);
                            double w1 = confusionMatrix.get(sourceDepLabel+"-"+targetDepLabel);
                            double w2 = confusionMatrix.get(targetDepLabel+"-"+sourceDepLabel);
                            learningWeight = (w1 + w2)/2;
                        }

                        ap.learnInstance(featureVector, label, learningWeight);
                    }
                }
            }
            System.out.print(trainSentencesInCONLLFormat.size()+"\n\n");
            if (devSentencesInCONLLFormat.size()!=0){
                //making prediction on dev data using the model trained in this iter
                AveragedPerceptron decodeAp = ap.calculateAvgWeights();
                int correct = 0;
                int total = 0;

                for (int sIdx = 0; sIdx < devSentencesInCONLLFormat.size(); sIdx++) {
                    if (sIdx % 1000 ==0)
                        System.out.print(sIdx+"...");
                    Sentence sentence = new Sentence(devSentencesInCONLLFormat.get(sIdx), indexMap);
                    ArrayList<Integer> goldPredicateIndices = sentence.getPredicatesIndices();

                    for (int wordIdx = 1; wordIdx < sentence.getLength(); wordIdx++) {
                        total++;
                        Object[] featureVector = FeatureExtractor.extractPIFeatures(wordIdx, sentence, numOfPIFeatures, indexMap);
                        String goldLabel = (goldPredicateIndices.contains(wordIdx)) ? "1" : "0";
                        String prediction = decodeAp.predict(featureVector);
                        if (goldLabel.equals(prediction))
                            correct++;
                    }
                }
                System.out.print(devSentencesInCONLLFormat.size()+"\n");
                double acc = (double) correct / total;
                System.out.print("Accuracy: "+ acc +"\n");
                if (acc > bestAcc) {
                    noImprovement = 0;
                    bestAcc = acc;
                    System.out.print("\nSaving final model...");
                    ModelInfo.saveModel(ap, PIModelPath);
                    System.out.println("Done!");
                } else {
                    noImprovement++;
                    if (noImprovement > 2) {
                        System.out.print("\nEarly stopping...");
                        break;
                    }
                }
            }
        }
        if (devSentencesInCONLLFormat.size()==0){
            System.out.print("\nSaving final model...");
            ModelInfo.saveModel(ap, PIModelPath);
            System.out.println("Done!");
        }
    }

    public static void predict (ArrayList<String> evalSentencesInCONLLFormat, IndexMap indexMap,
                                String PIModelPath, int numOfPIFeatures, String path2SavePredictions) throws Exception {
        HashSet<Integer>[] PIPredictions = new HashSet[evalSentencesInCONLLFormat.size()];
        AveragedPerceptron classifier = IO.load(PIModelPath);

        for (int senIdx =0 ; senIdx < evalSentencesInCONLLFormat.size(); senIdx++){
            HashSet<Integer> prediction4ThisSentence = new HashSet<>();
            Sentence sentence = new Sentence(evalSentencesInCONLLFormat.get(senIdx), indexMap);

            for (int wordIdx =0; wordIdx< sentence.getLength(); wordIdx++){
                Object[] featureVector = FeatureExtractor.extractPIFeatures(wordIdx, sentence, numOfPIFeatures, indexMap);
                String prediction = classifier.predict(featureVector);
                if (prediction.equals("1"))
                    prediction4ThisSentence.add(wordIdx);
            }
            PIPredictions[senIdx] = prediction4ThisSentence;
        }
        IO.write(PIPredictions, path2SavePredictions);
    }

}