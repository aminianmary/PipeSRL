package SupervisedSRL.PD;

import Sentence.Sentence;
import Sentence.PA;
import Sentence.PAs;
import Sentence.Argument;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.IndexMap;
import jdk.nashorn.internal.runtime.ECMAException;
import ml.AveragedPerceptron;
import util.IO;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Maryam Aminian on 5/19/16.
 * Predicate disambiguation modules
 */
public class PD {

    public static void main(String[] args) throws Exception {

        String inputFile = args[0];
        String modelDir = args[1];

        final IndexMap indexMap = new IndexMap(inputFile);

        //read trainJoint and test sentences
        ArrayList<String> sentencesInCONLLFormat = IO.readCoNLLFile(inputFile);

        int totalNumOfSentences = sentencesInCONLLFormat.size();
        int trainSize = (int) Math.floor(0.8 * totalNumOfSentences);

        List<String> train = sentencesInCONLLFormat.subList(0, trainSize);
        List<String> test = sentencesInCONLLFormat.subList(trainSize, totalNumOfSentences);

        //training
        train(train, indexMap, 10, modelDir);

        //prediction
        HashMap<Integer, String>[] predictions= new HashMap[test.size()];
        System.out.println("Prediction started...");
        for (int senIdx=0; senIdx<test.size(); senIdx++) {
            boolean decode = true;
            Sentence sentence = new Sentence(test.get(senIdx), indexMap, decode);
            predictions[senIdx] = predict(sentence, indexMap, modelDir);
        }

    }


    public static void train (List<String> trainSentencesInCONLLFormat, IndexMap indexMap, int numberOfTrainingIterations, String modelDir)
            throws Exception
    {
    int pdFeatSize =11;
        //creates lexicon of all predicates in the trainJoint set
        HashMap<Integer,  HashMap<Integer, HashSet<pLexiconEntry>>> trainPLexicon =
                buildPredicateLexicon(trainSentencesInCONLLFormat, indexMap);

        System.out.println("Training Started...");

        for (int plem: trainPLexicon.keySet())
        {
            //extracting feature vector for each training example
            for (int ppos: trainPLexicon.get(plem).keySet())
            {
                HashSet<pLexiconEntry> featVectors= trainPLexicon.get(plem).get(ppos);
                HashSet<String> labelSet= getLabels (featVectors);

                AveragedPerceptron ap = new AveragedPerceptron(labelSet, pdFeatSize);

                //System.out.print("training model for predicate/pos -->"+ plem+"|"+ppos+"\n");
                for (int i=0; i< numberOfTrainingIterations; i++)
                {
                    //System.out.print("iteration:" + i + "...");
                    for (pLexiconEntry ple: trainPLexicon.get(plem).get(ppos))
                    {
                        //trainJoint average perceptron
                        String plabel= ple.getPlabel();

                        ap.learnInstance(ple.getPdfeats(), plabel);
                    }
                }

                //System.out.print("\nSaving model...");
                ap.saveModel(modelDir+"/"+plem+"_"+ppos);
                //System.out.println("Done!");

            }
        }
        System.out.println("Done!");
    }


    public static HashMap<Integer, String> predict (Sentence sentence, IndexMap indexMap, String modelDir) throws Exception {
        File f1;
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] sentenceLemmas = sentence.getLemmas();
        int[] sentencePOSTags = sentence.getPosTags();
        HashMap<Integer, String> predictions = new HashMap<Integer, String>();
        for (PA pa : pas) {
            int pIdx = pa.getPredicateIndex();
            String pLabel = pa.getPredicateLabel();
            int plem = sentenceLemmas[pIdx];
            int ppos = sentencePOSTags[pIdx];
            Object[] pdfeats = FeatureExtractor.extractFeatures(pIdx, pLabel , -1, sentence, "PD", 11, indexMap);
            f1 = new File(modelDir + "/" + plem + "_" + ppos);
            if (f1.exists() && !f1.isDirectory()) {
                AveragedPerceptron classifier = AveragedPerceptron.loadModel(modelDir + "/" + plem + "_" + ppos);
                String prediction = classifier.predict(pdfeats);
                predictions.put(pIdx, prediction);
            }
        }
        return predictions;
    }



    public static HashMap<Integer,  HashMap<Integer, HashSet<pLexiconEntry>>> buildPredicateLexicon
            (List<String> sentencesInCONLLFormat, IndexMap indexMap)
    {
        HashMap<Integer,  HashMap<Integer, HashSet<pLexiconEntry>>> pLexicon=
                new HashMap<Integer, HashMap<Integer, HashSet<pLexiconEntry>>>();

        boolean decode = false;
        for (int senID=0; senID< sentencesInCONLLFormat.size(); senID++)
        {
            Sentence sentence= new Sentence(sentencesInCONLLFormat.get(senID), indexMap, decode);

            ArrayList<PA> pas= sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            int[] sentenceLemmas= sentence.getLemmas();
            int[] sentencePOSTags= sentence.getPosTags();

            for (PA pa:pas)
            {
                int pIdx= pa.getPredicateIndex();
                int plem= sentenceLemmas[pIdx];
                String plabel= pa.getPredicateLabel();
                int ppos= sentencePOSTags[pIdx];

                Object[] pdfeats = FeatureExtractor.extractFeatures(pIdx, plabel, -1 ,sentence, "PD", 11, indexMap);
                pLexiconEntry ple= new pLexiconEntry(plabel, pdfeats);

                if (!pLexicon.containsKey(plem))
                {
                    HashMap<Integer, HashSet<pLexiconEntry>> posDic= new  HashMap<Integer, HashSet<pLexiconEntry>>();
                    HashSet<pLexiconEntry> featVectors= new HashSet<pLexiconEntry>();
                    featVectors.add(ple);
                    posDic.put(ppos, featVectors);
                    pLexicon.put(plem, posDic);

                }
                else if (!pLexicon.get(plem).containsKey(ppos))
                {
                    HashSet<pLexiconEntry> featVectors= new HashSet<pLexiconEntry>();
                    featVectors.add(ple);
                    pLexicon.get(plem).put(ppos, featVectors);
                }else
                {
                    pLexicon.get(plem).get(ppos).add(ple);
                }

            }
        }

        return pLexicon;
    }



    public static HashSet<String> getLabels (HashSet<pLexiconEntry> featVectors)
    {
        HashSet<String> labelSet= new HashSet<String>();
        for (pLexiconEntry ple: featVectors)
            labelSet.add(ple.getPlabel());

        return labelSet;
    }

}
