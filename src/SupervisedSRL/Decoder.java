package SupervisedSRL;

import Sentence.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.*;
import de.bwaldvogel.liblinear.*;
import ml.AveragedPerceptron;
import ml.Adam;
import util.IO;

import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by Maryam Aminian on 5/24/16.
 */
public class Decoder {

    AveragedPerceptron aiClassifier; //argument identification (binary classifier)
    AveragedPerceptron acClassifier; //argument classification (multi-class classifier)
    Model aiClassifier_ll;
    Model acClassifier_ll;
    Adam aiClassifier_adam;
    Adam acClassifier_adam;

    public Decoder(AveragedPerceptron classifier, String state) {

        if (state.equals("AI")) {
            this.aiClassifier = classifier;
        } else if (state.equals("AC") || state.equals("joint")) {
            this.acClassifier = classifier;
        }
    }


    public Decoder(AveragedPerceptron aiClassifier, AveragedPerceptron acClassifier) {

        this.aiClassifier = aiClassifier;
        this.acClassifier = acClassifier;
    }


    public Decoder(Model aiClassifier, Model acClassifier) {
        this.aiClassifier_ll = aiClassifier;
        this.acClassifier_ll = acClassifier;
    }


    public Decoder(Adam aiClassifier, Adam acClassifier) {
        this.aiClassifier_adam = aiClassifier;
        this.acClassifier_adam = acClassifier;
    }


    public Decoder(Model classifier, String state) {

        if (state.equals("AI")) {
            this.aiClassifier_ll = classifier;
        } else if (state.equals("AC") || state.equalsIgnoreCase("joint")) {
            this.acClassifier_ll = classifier;
        }
    }


    public Decoder(Adam classifier, String state) {

        if (state.equals("AI")) {
            this.aiClassifier_adam = classifier;
        } else if (state.equals("AC") || state.equalsIgnoreCase("joint")) {
            this.acClassifier_adam = classifier;
        }
    }

    ////////////////////////////////// DECODE ////////////////////////////////////////////////////////

    //stacked decoding
    public static void decode(Decoder decoder, IndexMap indexMap, String devDataPath, String[] labelMap,
                              int aiMaxBeamSize, int acMaxBeamSize,
                              int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures,
                              String modelDir, String outputFile,
                              HashMap<Object, Integer>[] aiFeatDict,
                              HashMap<Object, Integer>[] acFeatDict,
                              ClassifierType classifierType, boolean greedy) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");

        System.out.println("Decoding started (on dev data)...");
        long startTime = System.currentTimeMillis();
        boolean decode = true;
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devDataPath);
        TreeMap<Integer, Prediction>[] predictions = new TreeMap[devSentencesInCONLLFormat.size()];
        ArrayList<ArrayList<String>> sentencesToWriteOutputFile = new ArrayList<ArrayList<String>>();

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());

            String devSentence = devSentencesInCONLLFormat.get(d);
            Sentence sentence = new Sentence(devSentence, indexMap, decode);

            predictions[d] = decoder.predict(sentence, indexMap, aiMaxBeamSize, acMaxBeamSize,
                    numOfAIFeatures, numOfACFeatures, numOfPDFeatures, modelDir,aiFeatDict,acFeatDict, classifierType, greedy);

            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentence));
        }
        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, labelMap, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
    }

    //joint decoding
    public static void decode(Decoder decoder, IndexMap indexMap, String devData,
                              String[] labelMap,
                              int maxBeamSize, int numOfFeatures, int numOfPDFeatures,
                              String modelDir,
                              String outputFile,
                              HashMap<Object, Integer>[] featDict,
                              ClassifierType classifierType, boolean greedy) throws Exception {

        DecimalFormat format = new DecimalFormat("##.00");

        System.out.println("Decoding started (on dev data)...");
        long startTime = System.currentTimeMillis();
        boolean decode = true;
        List<String> devSentencesInCONLLFormat = IO.readCoNLLFile(devData);
        TreeMap<Integer, Prediction>[] predictions = new TreeMap[devSentencesInCONLLFormat.size()];
        ArrayList<ArrayList<String>> sentencesToWriteOutputFile = new ArrayList<ArrayList<String>>();

        for (int d = 0; d < devSentencesInCONLLFormat.size(); d++) {
            if (d % 1000 == 0)
                System.out.println(d + "/" + devSentencesInCONLLFormat.size());
            String devSentence = devSentencesInCONLLFormat.get(d);
            Sentence sentence = new Sentence(devSentence, indexMap, decode);
            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentence));

            predictions[d] = decoder.predictJoint(sentence, indexMap, maxBeamSize, numOfFeatures, numOfPDFeatures, modelDir,featDict, classifierType, greedy);
        }

        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, labelMap, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
    }


    ////////////////////////////////// PREDICT ////////////////////////////////////////////////////////

    public HashMap<Integer, Prediction> predictAI(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                                                  int numOfFeatures, String modelDir, int numOfPDFeatures,
                                                  HashMap<Object, Integer>[] featDict,
                                                  ClassifierType classifierType, boolean greedy)
            throws Exception {
        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);
        HashMap<Integer, Prediction> predictedPAs = new HashMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = new ArrayList();
            ArrayList<Integer> aiCandidatesGreedy= new ArrayList<Integer>();
            HashMap<Integer, Integer> highestScorePrediction= new HashMap<Integer, Integer>();

            if (!greedy)
            {
                aiCandidates = getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfFeatures, featDict, classifierType);
                highestScorePrediction= getHighestScorePredication(aiCandidates);

            }else {
                aiCandidatesGreedy = getBestAICandidatesGreedy(sentence, pIdx, indexMap, numOfFeatures, featDict, classifierType);
                for (int idx=0; idx< aiCandidatesGreedy.size(); idx++) {
                    int wordIdx = aiCandidatesGreedy.get(idx);
                    highestScorePrediction.put(wordIdx, 1);
                }
            }

            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }



    public TreeMap<Integer, Prediction> predict(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                                                int acMaxBeamSize, int numOfAIFeatures, int numOfACFeatures,
                                                int numOfPDFeatures, String modelDir,
                                                HashMap<Object, Integer>[] aiFeatDict,
                                                HashMap<Object, Integer>[] acFeatDict,
                                                ClassifierType classifierType,
                                                boolean greedy) throws Exception {


        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);
        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates= new ArrayList();
            ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates= new ArrayList();

            ArrayList<Integer> aiCandidatesGreedy = new ArrayList<Integer>();
            ArrayList<Integer> acCandidatesGreedy = new ArrayList<Integer>();
            HashMap<Integer, Integer> highestScorePrediction = new HashMap<Integer, Integer>();

            if (!greedy)
            {
                aiCandidates = getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfAIFeatures, aiFeatDict, classifierType);
                acCandidates = getBestACCandidates(sentence, pIdx, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures, acFeatDict, classifierType);
                highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates);
            }else
            {
                aiCandidatesGreedy = getBestAICandidatesGreedy(sentence, pIdx, indexMap, numOfAIFeatures, aiFeatDict, classifierType);
                acCandidatesGreedy = getBestACCandidatesGreedy(sentence, pIdx, indexMap, aiCandidatesGreedy, numOfACFeatures, acFeatDict, classifierType);
                for (int idx=0; idx< acCandidatesGreedy.size(); idx++) {
                    int wordIdx = aiCandidatesGreedy.get(idx);
                    int label = acCandidatesGreedy.get(idx);
                    highestScorePrediction.put(wordIdx, label);
                }
            }
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }


    //this function is used for joint ai-ac decoding
    public TreeMap<Integer, Prediction> predictJoint(Sentence sentence, IndexMap indexMap,
                                                     int maxBeamSize, int numOfFeatures, int numOfPDFeatures,
                                                     String modelDir, HashMap<Object, Integer>[] featDict,
                                                     ClassifierType classifierType,boolean greedy) throws Exception {

        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);
        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> candidates= new ArrayList();
            int[] candidatesGreedy = new int[sentence.getWords().length];
            HashMap<Integer, Integer> highestScorePrediction= new HashMap<Integer, Integer>();

            if (!greedy) {
                candidates = getBestJointCandidates(sentence, pIdx, indexMap, maxBeamSize, numOfFeatures, featDict, classifierType);
                highestScorePrediction = getHighestScorePredicationJoint(candidates, pIdx);
            }
            else {
                candidatesGreedy = getBestJointCandidatesGreedy(sentence, pIdx, indexMap, numOfFeatures, featDict, classifierType);
                for (int idx=0; idx< candidatesGreedy.length; idx++)
                {
                    highestScorePrediction.put(idx, candidatesGreedy[idx]);
                }
            }
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }


    ////////////////////////////////// GET BEST CANDIDATES ///////////////////////////////////////////////

    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidates
            (Sentence sentence, int pIdx, IndexMap indexMap, int maxBeamSize, int numOfFeatures,
             HashMap<Object, Integer>[] featDict, ClassifierType classifierType) throws Exception {
        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));

        int[] sentenceWords = sentence.getWords();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
            double score0 = Double.POSITIVE_INFINITY;
            double score1 = Double.NEGATIVE_INFINITY;

            if (classifierType == ClassifierType.AveragedPerceptron) {
                double[] scores = aiClassifier.score(featVector);
                score0 = scores[0];
                score1 = scores[1];
            }else if (classifierType == ClassifierType.Liblinear) {
                ArrayList<FeatureNode> feats = new ArrayList<FeatureNode>();
                for (int d = 0; d < featVector.length; d++)
                    if (featDict[d].containsKey(featVector[d]))
                        //seen feature value
                        feats.add(new FeatureNode(featDict[d].get(featVector[d]), 1));
                    else
                        //unseen feature value
                        feats.add(new FeatureNode(featDict[d].get(Pipeline.unseenSymbol), 1));
                FeatureNode[] featureNodes = feats.toArray(new FeatureNode[0]);

                double[] probEstimates = new double[2];
                int prediction = (int) Linear.predictProbability(aiClassifier_ll, featureNodes, probEstimates);
                score0 = Math.log(probEstimates[0]);
                score1 = Math.log(probEstimates[1]);

            }else if (classifierType == ClassifierType.Adam){
                ArrayList<Integer> feats = new ArrayList<Integer>();
                for (int d = 0; d < featVector.length; d++)
                    if (featDict[d].containsKey(featVector[d]))
                        //seen feature value
                        feats.add(featDict[d].get(featVector[d]));
                    else
                        //unseen feature value
                        feats.add(featDict[d].get(Pipeline.unseenSymbol));

                double[] probEstimates = new double[2];
                int prediction= aiClassifier_adam.argmax(feats,probEstimates);

                score0 = Math.log(probEstimates[0]);
                score1 = Math.log(probEstimates[1]);
            }

            // build an intermediate beam
            TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

            for (int index = 0; index < currentBeam.size(); index++) {
                double currentScore = currentBeam.get(index).first;
                BeamElement be0 = new BeamElement(index, currentScore + score0, 0);
                BeamElement be1 = new BeamElement(index, currentScore + score1, 1);

                newBeamHeap.add(be0);
                if (newBeamHeap.size() > maxBeamSize)
                    newBeamHeap.pollFirst();

                newBeamHeap.add(be1);
                if (newBeamHeap.size() > maxBeamSize)
                    newBeamHeap.pollFirst();
            }

            ArrayList<Pair<Double, ArrayList<Integer>>> newBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>(maxBeamSize);
            for (BeamElement beamElement : newBeamHeap) {
                ArrayList<Integer> newArrayList = new ArrayList<Integer>();
                for (int b : currentBeam.get(beamElement.index).second)
                    newArrayList.add(b);
                if (beamElement.label == 1)
                    newArrayList.add(wordIdx);
                newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score, newArrayList));
            }

            // replace the old beam with the intermediate beam
            currentBeam = newBeam;
        }

        return currentBeam;
    }



    //getting highest score AI candidate (AP/LL/Adam) without Beam Search
    private ArrayList<Integer> getBestAICandidatesGreedy
            (Sentence sentence, int pIdx, IndexMap indexMap, int numOfFeatures, HashMap<Object, Integer>[] featDict,
             ClassifierType classifierType) throws Exception {
        int[] sentenceWords = sentence.getWords();
        ArrayList<Integer> aiCandids = new ArrayList<Integer>();
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            if (classifierType == ClassifierType.AveragedPerceptron) {
                Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
                double score1 = aiClassifier.score(featVector)[1];
                if (score1 >= 0)
                    aiCandids.add(wordIdx);
            } else if (classifierType == ClassifierType.Liblinear) {
                Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
                ArrayList<FeatureNode> feats = new ArrayList<FeatureNode>();
                for (int d = 0; d < featVector.length; d++)
                    if (featDict[d].containsKey(featVector[d]))
                        //seen feature value
                        feats.add(new FeatureNode(featDict[d].get(featVector[d]), 1));
                    else
                        //unseen feature value
                        feats.add(new FeatureNode(featDict[d].get(Pipeline.unseenSymbol), 1));
                FeatureNode[] featureNodes = feats.toArray(new FeatureNode[0]);
                double[] probEstimates = new double[2];
                int prediction = (int) Linear.predictProbability(aiClassifier_ll, featureNodes, probEstimates);
                if (probEstimates[1] > probEstimates[0])
                    aiCandids.add(wordIdx);
            } else if (classifierType == ClassifierType.Adam) {

                Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
                ArrayList<Integer> feats = new ArrayList<Integer>();
                for (int d = 0; d < featVector.length; d++)
                    if (featDict[d].containsKey(featVector[d]))
                        //seen feature value
                        feats.add(featDict[d].get(featVector[d]));
                    else
                        //unseen feature value
                        feats.add(featDict[d].get(Pipeline.unseenSymbol));

                double[] probEstimates = new double[2];
                int prediction = aiClassifier_adam.argmax(feats, probEstimates);
                if (probEstimates[1] > probEstimates[0])
                    aiCandids.add(wordIdx);
            }
        }
        return aiCandids;
    }


    private ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getBestACCandidates
            (Sentence sentence, int pIdx, IndexMap indexMap,
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates, int maxBeamSize, int numOfFeatures,
             HashMap<Object, Integer>[] featDict, ClassifierType classifierType) throws Exception {

        int numOfLabels = 0;
        if (classifierType== ClassifierType.AveragedPerceptron)
            numOfLabels = acClassifier.getLabelMap().length;
        else if (classifierType == ClassifierType.Liblinear)
            numOfLabels = acClassifier_ll.getLabels().length;
        else if (classifierType == ClassifierType.Adam)
            numOfLabels = acClassifier_adam.getLabelMap().length;

        ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> finalACCandidates = new ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>>();

        for (Pair<Double, ArrayList<Integer>> aiCandidate : aiCandidates) {
            //for each AI candidate generated by aiClassifier
            double aiScore = aiCandidate.first;
            ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
            currentBeam.add(new Pair<Double, ArrayList<Integer>>(aiScore, new ArrayList<Integer>()));

            // Gradual building of the beam for the words identified as an argument by AI classifier
            for (int wordIdx : aiCandidate.second) {
                // retrieve candidates for the current word
                Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
                double[] labelScores = new double[numOfLabels];

                if (classifierType== ClassifierType.AveragedPerceptron) {
                    labelScores = acClassifier.score(featVector);
                }
                else if (classifierType == ClassifierType.Liblinear){
                    ArrayList<FeatureNode> feats = new ArrayList<FeatureNode>();
                    for (int d = 0; d < featVector.length; d++)
                        if (featDict[d].containsKey(featVector[d]))
                            //seen feature value
                            feats.add(new FeatureNode(featDict[d].get(featVector[d]), 1));
                        else
                            //unseen feature value
                            feats.add(new FeatureNode(featDict[d].get(Pipeline.unseenSymbol), 1));
                    FeatureNode[] featureNodes = feats.toArray(new FeatureNode[0]);

                    double[] probEstimates = new double[numOfLabels];
                    int prediction = (int) Linear.predictProbability(acClassifier_ll, featureNodes, probEstimates);
                    for (int labelIdx=0; labelIdx< numOfLabels; labelIdx++)
                        labelScores[labelIdx] = Math.log(probEstimates[labelIdx]);

                }else if (classifierType == ClassifierType.Adam){
                    ArrayList<Integer> feats = new ArrayList<Integer>();
                    for (int d = 0; d < featVector.length; d++)
                        if (featDict[d].containsKey(featVector[d]))
                            //seen feature value
                            feats.add(featDict[d].get(featVector[d]));
                        else
                            //unseen feature value
                            feats.add(featDict[d].get(Pipeline.unseenSymbol));

                    double[] probEstimates = new double[numOfLabels];
                    int prediction = acClassifier_adam.argmax(feats, probEstimates);
                    for (int labelIdx=0; labelIdx< numOfLabels; labelIdx++)
                        labelScores[labelIdx] = Math.log(probEstimates[labelIdx]);
                }

                // build an intermediate beam
                TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

                for (int index = 0; index < currentBeam.size(); index++) {
                    double currentScore = currentBeam.get(index).first;

                    for (int labelIdx = 0; labelIdx < numOfLabels; labelIdx++) {
                        newBeamHeap.add(new BeamElement(index, currentScore + labelScores[labelIdx], labelIdx));
                        if (newBeamHeap.size() > maxBeamSize)
                            newBeamHeap.pollFirst();
                    }
                }

                ArrayList<Pair<Double, ArrayList<Integer>>> newBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>(maxBeamSize);

                for (BeamElement beamElement : newBeamHeap) {
                    ArrayList<Integer> newArrayList = new ArrayList<Integer>();
                    for(int b:currentBeam.get(beamElement.index).second)
                        newArrayList.add(b);
                    newArrayList.add(beamElement.label);
                    newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score, newArrayList));
                }

                // replace the old beam with the intermediate beam
                currentBeam = newBeam;
            }

            //current beam for this ai candidates is built
            finalACCandidates.add(currentBeam);
        }

        return finalACCandidates;
    }


    //getting highest score AC candidate (AP/LL/Adam) without Beam Search
    private ArrayList<Integer> getBestACCandidatesGreedy
            (Sentence sentence, int pIdx, IndexMap indexMap, ArrayList<Integer> aiCandidates, int numOfFeatures,
             HashMap<Object, Integer>[] featDict, ClassifierType classifierType) throws Exception {

        ArrayList<Integer> acCandids = new ArrayList<Integer>();
        for (int aiCandidIdx = 0; aiCandidIdx < aiCandidates.size(); aiCandidIdx++) {
            int wordIdx = aiCandidates.get(aiCandidIdx);
            if (classifierType == ClassifierType.AveragedPerceptron)
            {
                Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
                double[] labelScores = acClassifier.score(featVector);
                int predictedLabel = argmax(labelScores);
                acCandids.add(predictedLabel);
            }else if (classifierType == ClassifierType.Liblinear)
            {
                // retrieve candidates for the current word
                int[] labels = acClassifier_ll.getLabels();
                Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
                ArrayList<FeatureNode> feats = new ArrayList<FeatureNode>();
                for (int d = 0; d < featVector.length; d++)
                    if (featDict[d].containsKey(featVector[d]))
                        //seen feature value
                        feats.add(new FeatureNode(featDict[d].get(featVector[d]), 1));
                    else
                        //unseen feature value
                        feats.add(new FeatureNode(featDict[d].get(Pipeline.unseenSymbol), 1));
                FeatureNode[] featureNodes = feats.toArray(new FeatureNode[0]);

                double[] probEstimates = new double[labels.length];
                int prediction = (int) Linear.predictProbability(acClassifier_ll, featureNodes, probEstimates);
                acCandids.add(prediction);
            }else if (classifierType == ClassifierType.Adam) {
                String[] labelMap = acClassifier_adam.getLabelMap();
                // retrieve candidates for the current word
                Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
                ArrayList<Integer> feats = new ArrayList<Integer>();
                for (int d = 0; d < featVector.length; d++)
                    if (featDict[d].containsKey(featVector[d]))
                        //seen feature value
                        feats.add(featDict[d].get(featVector[d]));
                    else
                        //unseen feature value
                        feats.add(featDict[d].get(Pipeline.unseenSymbol));

                double[] probEstimates = new double[labelMap.length];
                int prediction = acClassifier_adam.argmax(feats, probEstimates);
                acCandids.add(prediction);
            }
        }
        assert aiCandidates.size() == acCandids.size();
        return acCandids;
    }


    //this function is used for joint ai-ac decoding (AP/LL/Adam)
    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestJointCandidates
    (Sentence sentence, int pIdx, IndexMap indexMap,
     int maxBeamSize, int numOfFeatures, HashMap<Object, Integer>[] featDict, ClassifierType classifierType) throws Exception {

        int numOfLabels = 0;
        if (classifierType == ClassifierType.AveragedPerceptron)
            numOfLabels = acClassifier.getLabelMap().length;
        else if (classifierType == ClassifierType.Liblinear)
            numOfLabels = acClassifier_ll.getLabels().length;
        else if (classifierType == ClassifierType.Adam)
            numOfLabels = acClassifier_adam.getLabelMap().length;

        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));


        // Gradual building of the beam for all words in the sentence
        for (int wordIdx = 1; wordIdx < sentence.getWords().length; wordIdx++) {
            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
            double[] labelScores= new double[numOfLabels];

            if (classifierType == ClassifierType.AveragedPerceptron)
                labelScores = acClassifier.score(featVector);

            else if (classifierType == ClassifierType.Liblinear){
                ArrayList<FeatureNode> feats = new ArrayList<FeatureNode>();
                for (int d = 0; d < featVector.length; d++)
                    if (featDict[d].containsKey(featVector[d]))
                        //seen feature value
                        feats.add(new FeatureNode(featDict[d].get(featVector[d]), 1));
                    else
                        //unseen feature value
                        feats.add(new FeatureNode(featDict[d].get(Pipeline.unseenSymbol), 1));
                FeatureNode[] featureNodes = feats.toArray(new FeatureNode[0]);

                double[] probEstimates = new double[numOfLabels];
                int prediction = (int) Linear.predictProbability(acClassifier_ll, featureNodes, probEstimates);
                for (int labelIdx=0; labelIdx< numOfLabels; labelIdx++)
                    labelScores[labelIdx] = Math.log(probEstimates[labelIdx]);

            }else if (classifierType == ClassifierType.Adam){
                ArrayList<Integer> feats = new ArrayList<Integer>();
                for (int d = 0; d < featVector.length; d++)
                    if (featDict[d].containsKey(featVector[d]))
                        //seen feature value
                        feats.add(featDict[d].get(featVector[d]));
                    else
                        //unseen feature value
                        feats.add(featDict[d].get(Pipeline.unseenSymbol));
                double[] probEstimates = new double[numOfLabels];
                int prediction = acClassifier_adam.argmax(feats, probEstimates);
                for (int labelIdx=0; labelIdx< numOfLabels; labelIdx++)
                    labelScores[labelIdx] = Math.log(probEstimates[labelIdx]);
            }

            // build an intermediate beam
            TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

            for (int index = 0; index < currentBeam.size(); index++) {
                double currentScore = currentBeam.get(index).first;

                for (int labelIdx = 0; labelIdx < numOfLabels; labelIdx++) {
                    newBeamHeap.add(new BeamElement(index, currentScore + labelScores[labelIdx], labelIdx));
                    if (newBeamHeap.size() > maxBeamSize)
                        newBeamHeap.pollFirst();
                }
            }

            ArrayList<Pair<Double, ArrayList<Integer>>> newBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>(maxBeamSize);

            for (BeamElement beamElement : newBeamHeap) {
                ArrayList<Integer> newArrayList = (ArrayList<Integer>) currentBeam.get(beamElement.index).second.clone();
                newArrayList.add(beamElement.label);
                newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score, newArrayList));
            }
            // replace the old beam with the intermediate beam
            currentBeam = newBeam;
        }

        return currentBeam;
    }


    private int[] getBestJointCandidatesGreedy
            (Sentence sentence, int pIdx, IndexMap indexMap,
             int numOfFeatures, HashMap<Object, Integer>[] featDict, ClassifierType classifierType) throws Exception {

        int numOfLabels = 0;
        int[] predictedLabels = new int[sentence.getWords().length]; //for each word in the sentence, we have a label (either zero or non-zero)
        predictedLabels[0] = -1; //label for root element

        if (classifierType == ClassifierType.AveragedPerceptron)
            numOfLabels = acClassifier.getLabelMap().length;
        else if (classifierType == ClassifierType.Liblinear)
            numOfLabels = acClassifier_ll.getLabels().length;
        else if (classifierType == ClassifierType.Adam)
            numOfLabels = acClassifier_adam.getLabelMap().length;

        for (int wordIdx = 1; wordIdx < sentence.getWords().length; wordIdx++) {
            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
            double[] labelScores = new double[numOfLabels];

            if (classifierType == ClassifierType.AveragedPerceptron)
                labelScores = acClassifier.score(featVector);

            else if (classifierType == ClassifierType.Liblinear) {
                ArrayList<FeatureNode> feats = new ArrayList<FeatureNode>();
                for (int d = 0; d < featVector.length; d++)
                    if (featDict[d].containsKey(featVector[d]))
                        //seen feature value
                        feats.add(new FeatureNode(featDict[d].get(featVector[d]), 1));
                    else
                        //unseen feature value
                        feats.add(new FeatureNode(featDict[d].get(Pipeline.unseenSymbol), 1));
                FeatureNode[] featureNodes = feats.toArray(new FeatureNode[0]);

                double[] probEstimates = new double[numOfLabels];
                int prediction = (int) Linear.predictProbability(acClassifier_ll, featureNodes, probEstimates);
                for (int labelIdx = 0; labelIdx < numOfLabels; labelIdx++)
                    labelScores[labelIdx] = Math.log(probEstimates[labelIdx]);

            } else if (classifierType == ClassifierType.Adam) {
                ArrayList<Integer> feats = new ArrayList<Integer>();
                for (int d = 0; d < featVector.length; d++)
                    if (featDict[d].containsKey(featVector[d]))
                        //seen feature value
                        feats.add(featDict[d].get(featVector[d]));
                    else
                        //unseen feature value
                        feats.add(featDict[d].get(Pipeline.unseenSymbol));
                double[] probEstimates = new double[numOfLabels];
                int prediction = acClassifier_adam.argmax(feats, probEstimates);
                for (int labelIdx = 0; labelIdx < numOfLabels; labelIdx++)
                    labelScores[labelIdx] = Math.log(probEstimates[labelIdx]);
            }

            int prediction = argmax(labelScores);
            predictedLabels[wordIdx] = prediction;
        }
        return predictedLabels;
    }


    private HashMap<Integer, Integer> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates) {

        double highestScore = Double.NEGATIVE_INFINITY;
        ArrayList<Integer> highestScoreACSeq = new ArrayList<Integer>();
        int highestScoreSeqAIIndex = -1;
        int bestAIIndex = 0;
        int bestACIndex = 0;

        for (int aiIndex = 0; aiIndex < aiCandidates.size(); aiIndex++) {
            for (int acIndex = 0; acIndex < acCandidates.get(aiIndex).size(); acIndex++) {
                Pair<Double, ArrayList<Integer>> ar = acCandidates.get(aiIndex).get(acIndex);
                double score = ar.first;
                if (score > highestScore) {
                    highestScore = score;
                    bestAIIndex = aiIndex;
                    bestACIndex = acIndex;
                }
            }
        }

        //after finding highest score sequence in the list of AC candidates
        HashMap<Integer, Integer> wordIndexLabelMap = new HashMap<Integer, Integer>();

        ArrayList<Integer> acResult = acCandidates.get(bestAIIndex).get(bestACIndex).second;
        ArrayList<Integer> aiResult = aiCandidates.get(bestAIIndex).second;
        assert acResult.size() == aiResult.size();

        for (int i = 0; i < acResult.size(); i++)
            wordIndexLabelMap.put(aiResult.get(i), acResult.get(i));
        return wordIndexLabelMap;
    }


    private HashMap<Integer, Integer> getHighestScorePredication
            (ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates) {

        TreeSet<Pair<Double, ArrayList<Integer>>> sortedCandidates = new TreeSet<Pair<Double, ArrayList<Integer>>>(aiCandidates);
        Pair<Double, ArrayList<Integer>> highestScorePair = sortedCandidates.pollLast();

        //after finding highest score sequence in the list of candidates
        HashMap<Integer, Integer> wordIndexLabelMap = new HashMap<Integer, Integer>();
        ArrayList<Integer> highestScoreSeq = highestScorePair.second;

        for (int index : highestScoreSeq) {
            wordIndexLabelMap.put(index, 1);
        }

        return wordIndexLabelMap;
    }


    //this function is used for joint ai-ac modules
    private HashMap<Integer, Integer> getHighestScorePredicationJoint
    (ArrayList<Pair<Double, ArrayList<Integer>>> candidates, int pIndex) {

        TreeSet<Pair<Double, ArrayList<Integer>>> acCandidates4ThisSeq = new TreeSet<Pair<Double, ArrayList<Integer>>>(candidates);
        Pair<Double, ArrayList<Integer>> highestScorePair = acCandidates4ThisSeq.pollLast();

        //after finding highest score sequence in the list of candidates
        HashMap<Integer, Integer> wordIndexLabelMap = new HashMap<Integer, Integer>();
        ArrayList<Integer> highestScoreSeq = highestScorePair.second;

        int realIndex = 1;
        for (int k = 0; k < highestScoreSeq.size(); k++) {
            if (realIndex == pIndex)
                realIndex++;
            wordIndexLabelMap.put(realIndex, highestScoreSeq.get(k));
            realIndex++;
        }

        return wordIndexLabelMap;
    }


    private int argmax(double[] scores) {
        int argmax = -1;
        double max = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > max) {
                argmax = i;
                max = scores[i];
            }
        }
        return argmax;
    }

}
