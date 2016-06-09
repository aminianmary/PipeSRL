package SupervisedSRL.PD;

import Sentence.Sentence;
import Sentence.PA;
import Sentence.Argument;
import SupervisedSRL.Features.FeatureExtractor;
import jdk.nashorn.internal.runtime.ECMAException;
import ml.AveragedPerceptron;
import util.IO;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutionException;

/**
 * Created by Maryam Aminian on 5/19/16.
 * Predicate disambiguation modules
 */
public class PD {

    public static void main(String[] args) throws Exception{

        String inputFile= args[0];
        String modelDir= args[1];

        /*
        try
        {
        */
            //read train and test sentences
            ArrayList<String> sentencesInCONLLFormat  = IO.readCoNLLFile(inputFile);

            int totalNumOfSentences= sentencesInCONLLFormat.size();
            int trainSize= (int) Math.floor(0.8 * totalNumOfSentences);

            List<String> train= sentencesInCONLLFormat.subList(0, trainSize);
            List<String> test= sentencesInCONLLFormat.subList(trainSize, totalNumOfSentences);

            //training
            train(train, 10, modelDir);

            //prediction
            predict(test, modelDir);


        /*
        }catch (Exception e)
        {
            e.printStackTrace();
        }
        */
    }


    public static void train (List<String> trainSentencesInCONLLFormat,int numberOfTrainingIterations, String modelDir)
            throws Exception
    {

        //creates lexicon of all predicates in the train set
        HashMap<String,  HashMap<String, HashSet<pLexiconEntry>>> trainPLexicon =
                buildPredicateLexicon(trainSentencesInCONLLFormat);

        System.out.println("Training Started...");

        for (String plem: trainPLexicon.keySet())
        {
            //extracting feature vector for each training example
            for (String ppos: trainPLexicon.get(plem).keySet())
            {
                HashSet<pLexiconEntry> featVectors= trainPLexicon.get(plem).get(ppos);
                HashSet<String> labelSet= getLabels (featVectors);

                AveragedPerceptron ap = new AveragedPerceptron(labelSet);

                //System.out.print("training model for predicate/pos -->"+ plem+"|"+ppos+"\n");
                for (int i=0; i< numberOfTrainingIterations; i++)
                {
                    //System.out.print("iteration:" + i + "...");
                    for (pLexiconEntry ple: trainPLexicon.get(plem).get(ppos))
                    {
                        //train average perceptron
                        List<String> featVecs= Arrays.asList(ple.getPdfeats());
                        String plabel= ple.getPlabel();

                        ap.learnInstance(featVecs, plabel);
                    }
                }

                //System.out.print("\nSaving model...");
                ap.saveModel(modelDir+"/"+plem+"_"+ppos);
                //System.out.println("Done!");

            }
        }
        System.out.println("Done!");
    }


    public static void predict (List<String> testSentencesInCONLLFormat, String modelDir) throws Exception
    {

        HashMap<String,  HashMap<String, HashSet<pLexiconEntry>>> testPLexicon =
                buildPredicateLexicon(testSentencesInCONLLFormat);

        int correct=0;
        int numOfTestExamples=0;
        int numOfUnseenPredicates=0;

        System.out.println("Prediction started...");
        File f1;

        for (String plem: testPLexicon.keySet())
        {
            for (String ppos: testPLexicon.get(plem).keySet())
            {
                HashSet<pLexiconEntry> featVecs= testPLexicon.get(plem).get(ppos);
                for (pLexiconEntry ple: featVecs)
                {
                    numOfTestExamples++;

                    List<String> featVec= Arrays.asList(ple.getPdfeats());
                    String plabel = ple.getPlabel();

                    //System.out.println("Loading model for "+plem+"_"+ppos+"...");
                    f1= new File(modelDir+"/"+plem+"_"+ppos);

                    if (f1.exists() && !f1.isDirectory())
                    {
                        AveragedPerceptron classifier = AveragedPerceptron.loadModel(modelDir + "/" + plem + "_" + ppos);
                        String prediction = classifier.predict(featVec);
                        if (prediction.equals(plabel))
                            correct++;
                        //else
                            //System.out.println(plem + "\t" + ppos + "->" + prediction);
                    }else {
                        //System.out.println("model file does not exist for " + plem + "_" + ppos + "\nThus, predicate won't recieve a label");
                        numOfUnseenPredicates++;
                    }
                }
            }
        }

        System.out.println("Prediction Done!");

        System.out.println(correct+"\t"+numOfTestExamples);
        System.out.println(100*(float)correct/numOfTestExamples+"%");

        System.out.println("num Of Unseen Predicates: "+ numOfUnseenPredicates);

    }



    public static HashMap<String,  HashMap<String, HashSet<pLexiconEntry>>> buildPredicateLexicon (List<String> sentencesInCONLLFormat)
    {
        HashMap<String,  HashMap<String, HashSet<pLexiconEntry>>> pLexicon=
                new HashMap<String,  HashMap<String, HashSet<pLexiconEntry>>>();


        for (int senID=0; senID< sentencesInCONLLFormat.size(); senID++)
        {
            Sentence sentence= new Sentence(sentencesInCONLLFormat.get(senID));

            ArrayList<PA> pas= sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            String[] sentenceLemmas= sentence.getLemmas();
            String[] sentencePOSTags= sentence.getPosTags();

            for (PA pa:pas)
            {
                int pIdx= pa.getPredicateIndex();
                String plem= sentenceLemmas[pIdx];
                String plabel= pa.getPredicateLabel();
                String ppos= sentencePOSTags[pIdx];

                String[] pdfeats = FeatureExtractor.extractFeatures(pa.getPredicate(), -1 ,sentence, "PD", 11);
                pLexiconEntry ple= new pLexiconEntry(plabel, pdfeats);

                if (!pLexicon.containsKey(plem))
                {
                    HashMap<String, HashSet<pLexiconEntry>> posDic= new  HashMap<String, HashSet<pLexiconEntry>>();
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
