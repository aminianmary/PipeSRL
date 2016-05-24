package SupervisedSRL;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import Sentence.Sentence;
import Sentence.Predicate;
import Sentence.Argument;
import Sentence.PA;
import SupervisedSRL.Features.FeatureExtractor;
import ml.AveragedPerceptron;

/**
 * Created by Maryam Aminian on 5/23/16.
 */
public class aiTrain {

    public static void main(String[] args) {

    }

    public void train (List<String> trainSentencesInCONLLFormat,int numberOfTrainingIterations, String modelDir)
            throws Exception
    {
        HashSet<String> labelSet= new HashSet<String>();
        labelSet.add("1");
        labelSet.add("0");

        AveragedPerceptron ap = new AveragedPerceptron(labelSet);

        //building train instances
        Object[] instances = obtainTrainInstances(trainSentencesInCONLLFormat, "AI", 32);

        ArrayList<List<String>> featVectors= (ArrayList<List<String>>) instances[0] ;
        ArrayList<String> labels= (ArrayList<String>) instances[1];

        //training average perceptron
        for (int iter=0; iter< numberOfTrainingIterations; iter++)
        {
            System.out.print("iteration:" + iter + "...");
            for (int d=0; d< featVectors.size(); d++)
            {
                ap.learnInstance(featVectors.get(d), labels.get(d));
            }
        }

        System.out.print("\nSaving model...");
        ap.saveModel(modelDir+"/AI.model");
        System.out.println("Done!");

    }

    public Object[] obtainTrainInstances(List<String> sentencesInCONLLFormat, String state, int numOfFeatures)
    {
        ArrayList<List<String>> featVectors= new ArrayList<List<String>>();
        ArrayList<String> labels= new ArrayList<String>();

        for (String sentenceInCONLLFormat: sentencesInCONLLFormat)
        {
            Sentence sentence= new Sentence(sentenceInCONLLFormat);
            ArrayList<PA> pas= sentence.getPredicateArguments().getPredicateArgumentsAsArray();
            String[] sentenceWords= sentence.getWords();

            for (PA pa: pas)
            {
                Predicate currentP= pa.getPredicate();
                ArrayList<Argument> currentArgs= pa.getArguments();

                for (int wordIdx=0; wordIdx< sentenceWords.length; wordIdx++)
                {
                    String[] featVector= FeatureExtractor.extractFeatures(currentP, wordIdx,
                            sentence, state, numOfFeatures);

                    String label= (isArgument(wordIdx, currentArgs).equals(""))? "0": "1";

                    featVectors.add(Arrays.asList(featVector));
                    labels.add(label);
                }
            }
        }

        return new Object[]{featVectors, labels};
    }


    public String isArgument(int wordIdx, ArrayList<Argument> currentArgs)
    {
        for (Argument arg: currentArgs)
            if (arg.getIndex()==wordIdx)
                return arg.getType();
        return "";
    }

}
