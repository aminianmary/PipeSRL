package SupervisedSRL;

import Sentence.*;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.Strcutures.BeamElement;
import SupervisedSRL.Strcutures.Pair;
import ml.AveragedPerceptron;

import java.util.*;

/**
 * Created by Maryam Aminian on 5/24/16.
 */
public class ArgumentDecoder {

    AveragedPerceptron aiClassifier; //argument identification (binary classifier)
    AveragedPerceptron acClassifier; //argument classification (multi-class classifier)

    public ArgumentDecoder(AveragedPerceptron aiClassifier, AveragedPerceptron acClassifier) {
        this.aiClassifier = aiClassifier;
        this.acClassifier = acClassifier;
    }

    private ArrayList<Pair<Double,ArrayList<Integer>>> getBestAICandidates
            (Sentence sentence, PA pa, int maxBeamSize) {
        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0.,new ArrayList<Integer>()));

        String[] sentenceWords = sentence.getWords();
        Predicate currentPr = pa.getPredicate();

        // Gradual building of the beam
        for (int wordIdx = 0; wordIdx < sentenceWords.length; wordIdx++) {
            if(wordIdx==currentPr.getIndex())
                continue;

            // retrieve candidates for the current word
            String[] featVector = FeatureExtractor.extractFeatures(currentPr, wordIdx, sentence, "AI", 32);
            List<String> features = Arrays.asList(featVector);
            double score0 =  aiClassifier.score(features, "0");
            double score1 =  aiClassifier.score(features,"1");

            // build an intermediate beam
            TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

            for(int index=0;index<currentBeam.size();index++) {
                double currentScore = currentBeam.get(index).first;
                BeamElement be0 = new BeamElement(index,currentScore+score0,0);
                BeamElement be1 = new BeamElement(index,currentScore+score1,1);

                newBeamHeap.add(be0);
                if(newBeamHeap.size()>maxBeamSize)
                    newBeamHeap.pollFirst();

                newBeamHeap.add(be1);
                if(newBeamHeap.size()>maxBeamSize)
                    newBeamHeap.pollFirst();
            }

            ArrayList<Pair<Double, ArrayList<Integer>>> newBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>(maxBeamSize);
            for(BeamElement beamElement:newBeamHeap) {
                //todo check if it works properly
                ArrayList<Integer> newArrayList = (ArrayList<Integer>)currentBeam.get(beamElement.index).second.clone();
                if(beamElement.label==1)
                    newArrayList.add(wordIdx);
                newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score,newArrayList));
            }

            // replace the old beam with the intermediate beam
            currentBeam = newBeam;
        }

        return currentBeam;
    }

    public void predict(Sentence sentence, int maxBeamSize) throws Exception {
        ArrayList<PA> pas = sentence.getPredicateArguments().getPredicateArgumentsAsArray();


        for (PA pa : pas) {
            //for each predicate
            ArrayList<Argument> currentArgs = pa.getArguments();

            // get best k argument assignment candidates
            ArrayList<Pair<Double,ArrayList<Integer>>> aiCandiates = getBestAICandidates(sentence,pa,maxBeamSize);

            // get best <=k.l best argument label assignment

            }
        }


}
