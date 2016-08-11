package SupervisedSRL;

import Sentence.Sentence;
import SupervisedSRL.Features.FeatureExtractor;
import SupervisedSRL.PD.PD;
import SupervisedSRL.Strcutures.*;
import de.bwaldvogel.liblinear.*;
import de.bwaldvogel.liblinear.Train;
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


    public static void decode(Decoder decoder, IndexMap indexMap, String devDataPath, String[] labelMap,
                              int aiMaxBeamSize, int acMaxBeamSize,
                              int numOfAIFeatures, int numOfACFeatures, int numOfPDFeatures,
                              String modelDir, String outputFile,
                              HashMap<Object, Integer>[] aiFeatDict,
                              HashMap<Object, Integer>[] acFeatDict,
                              ClassifierType classifierType) throws Exception {

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
                    numOfAIFeatures, numOfACFeatures, numOfPDFeatures, modelDir,aiFeatDict,acFeatDict, classifierType);

            sentencesToWriteOutputFile.add(IO.getSentenceForOutput(devSentence));
        }
        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, labelMap, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
    }


    public static void decode(Decoder decoder, IndexMap indexMap, String devData,
                              String[] labelMap,
                              int maxBeamSize, int numOfFeatures, int numOfPDFeatures,
                              String modelDir,
                              String outputFile,
                              HashMap<Object, Integer>[] featDict,
                              ClassifierType classifierType) throws Exception {

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

            predictions[d] = decoder.predictJoint(sentence, indexMap, maxBeamSize, numOfFeatures, numOfPDFeatures, modelDir,featDict, classifierType);
        }

        IO.writePredictionsInCoNLLFormat(sentencesToWriteOutputFile, predictions, labelMap, outputFile);
        long endTime = System.currentTimeMillis();
        System.out.println("Total time for decoding: " + format.format(((endTime - startTime) / 1000.0) / 60.0));
    }


    private static TreeSet<Pair<Double, ArrayList<Integer>>> convertArrayListOfPairs2TreeSetOfPairs(ArrayList<Pair<Double, ArrayList<Integer>>> arrayOfPairs) {
        TreeSet<Pair<Double, ArrayList<Integer>>> treeSetOfPairs = new TreeSet<Pair<Double, ArrayList<Integer>>>();
        for (Pair<Double, ArrayList<Integer>> pair : arrayOfPairs) {
            treeSetOfPairs.add(pair);
        }
        return treeSetOfPairs;
    }


    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidates
            (Sentence sentence, int pIdx, IndexMap indexMap, int maxBeamSize, int numOfFeatures) throws Exception {
        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));

        int[] sentenceWords = sentence.getWords();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {

            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
            double[] scores = aiClassifier.score(featVector);
            double score0 = scores[0];
            double score1 = scores[1];

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


    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidatesLiblinear
            (Sentence sentence, int pIdx, IndexMap indexMap, int maxBeamSize, int numOfFeatures,
             HashMap<Object, Integer>[] featDict) throws Exception

    {
        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));

        int[] sentenceWords = sentence.getWords();

        //Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {

            //retrieve candidates for the current word
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

            double score0 = Math.log(probEstimates[0]);
            double score1 = Math.log(probEstimates[1]);

            //build an intermediate beam
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
                ArrayList<Integer> newArrayList = (ArrayList<Integer>) currentBeam.get(beamElement.index).second.clone();
                if (beamElement.label == 1)
                    newArrayList.add(wordIdx);
                newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score, newArrayList));
            }

            // replace the old beam with the intermediate beam
            currentBeam = newBeam;
        }

        return currentBeam;
    }


    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestAICandidatesAdam
            (Sentence sentence, int pIdx, IndexMap indexMap, int maxBeamSize, int numOfFeatures,
             HashMap<Object, Integer>[] featDict) throws Exception

    {
        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));

        int[] sentenceWords = sentence.getWords();

        //Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {

            //retrieve candidates for the current word
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
            int prediction= aiClassifier_adam.argmax(feats,probEstimates);

            double score0 = Math.log(probEstimates[0]);
            double score1 = Math.log(probEstimates[1]);

            //build an intermediate beam
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
                ArrayList<Integer> newArrayList = (ArrayList<Integer>) currentBeam.get(beamElement.index).second.clone();
                if (beamElement.label == 1)
                    newArrayList.add(wordIdx);
                newBeam.add(new Pair<Double, ArrayList<Integer>>(beamElement.score, newArrayList));
            }

            // replace the old beam with the intermediate beam
            currentBeam = newBeam;
        }

        return currentBeam;
    }


    //getting highest score AI candidate without Beam Search
    private HashMap<Integer, Integer> getHighestScoreAISeq(Sentence sentence, int pIdx, IndexMap indexMap, int numOfFeatures) throws Exception {
        int[] sentenceWords = sentence.getWords();
        HashMap<Integer, Integer> highestScoreAISeq = new HashMap<Integer, Integer>();

        // Gradual building of the beam
        for (int wordIdx = 1; wordIdx < sentenceWords.length; wordIdx++) {
            if (wordIdx == pIdx)
                continue;

            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractAIFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
            double score1 = aiClassifier.score(featVector)[1];

            if (score1 >= 0) {
                highestScoreAISeq.put(wordIdx, 1);
            }
        }

        return highestScoreAISeq;
    }



    private ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getBestACCandidates
            (Sentence sentence, int pIdx, IndexMap indexMap,
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             int maxBeamSize, int numOfFeatures) throws Exception {
        String[] labelMap = acClassifier.getLabelMap();
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
                double[] labelScores = acClassifier.score(featVector);

                // build an intermediate beam
                TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

                for (int index = 0; index < currentBeam.size(); index++) {
                    double currentScore = currentBeam.get(index).first;

                    for (int labelIdx = 0; labelIdx < labelMap.length; labelIdx++) {
                        newBeamHeap.add(new BeamElement(index, currentScore + labelScores[labelIdx], labelIdx));
                        if (newBeamHeap.size() > maxBeamSize)
                            newBeamHeap.pollFirst();
                    }
                }

                ArrayList<Pair<Double, ArrayList<Integer>>> newBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>(maxBeamSize);

                for (BeamElement beamElement : newBeamHeap) {
                    //todo check if it works properly
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


    private ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getBestACCandidatesLiblinear
            (Sentence sentence, int pIdx, IndexMap indexMap,
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             int maxBeamSize, int numOfFeatures,
             HashMap<Object, Integer>[] featDict) throws Exception {
        int[] labels = acClassifier_ll.getLabels();
        ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> finalACCandidates =
                new ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>>();


        for (Pair<Double, ArrayList<Integer>> aiCandidate : aiCandidates) {
            //for each AI candidate generated by aiClassifier
            Double aiScore = aiCandidate.first;
            ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
            currentBeam.add(new Pair<Double, ArrayList<Integer>>(aiScore, new ArrayList<Integer>()));


            // Gradual building of the beam for the words identified as an argument by AI classifier
            for (int wordIdx : aiCandidate.second) {

                // retrieve candidates for the current word
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
                double[] labelScores = new double[labels.length];
                int prediction = (int) Linear.predictProbability(acClassifier_ll, featureNodes, probEstimates);
                for (int labelIdx=0; labelIdx< labels.length; labelIdx++)
                        labelScores[labelIdx] = Math.log(probEstimates[labelIdx]);

                // build an intermediate beam
                TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

                for (int index = 0; index < currentBeam.size(); index++) {
                    double currentScore = currentBeam.get(index).first;

                    for (int labelIdx = 0; labelIdx < labels.length; labelIdx++) {
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

            //current beam for this ai candidates is built
            finalACCandidates.add(currentBeam);
        }

        return finalACCandidates;
    }


    private ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getBestACCandidatesAdam
            (Sentence sentence, int pIdx, IndexMap indexMap,
             ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates,
             int maxBeamSize, int numOfFeatures,
             HashMap<Object, Integer>[] featDict) throws Exception {
        String[] labelMap = acClassifier_adam.getLabelMap();
        ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> finalACCandidates =
                new ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>>();


        for (Pair<Double, ArrayList<Integer>> aiCandidate : aiCandidates) {
            //for each AI candidate generated by aiClassifier
            Double aiScore = aiCandidate.first;
            ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
            currentBeam.add(new Pair<Double, ArrayList<Integer>>(aiScore, new ArrayList<Integer>()));


            // Gradual building of the beam for the words identified as an argument by AI classifier
            for (int wordIdx : aiCandidate.second) {

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
                double[] labelScores = new double[labelMap.length];
                int prediction = acClassifier_adam.argmax(feats, probEstimates);
                for (int labelIdx=0; labelIdx< labelMap.length; labelIdx++)
                    labelScores[labelIdx] = Math.log(probEstimates[labelIdx]);

                // build an intermediate beam
                TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

                for (int index = 0; index < currentBeam.size(); index++) {
                    double currentScore = currentBeam.get(index).first;

                    for (int labelIdx = 0; labelIdx < labelMap.length; labelIdx++) {
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

            //current beam for this ai candidates is built
            finalACCandidates.add(currentBeam);
        }

        return finalACCandidates;
    }


    //this function is used for joint ai-ac decoding
    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestCandidates
    (Sentence sentence, int pIdx, IndexMap indexMap,
     int maxBeamSize, int numOfFeatures) throws Exception {
        String[] labelMap = acClassifier.getLabelMap();

        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));


        // Gradual building of the beam for all words in the sentence
        for (int wordIdx = 1; wordIdx < sentence.getWords().length; wordIdx++) {
            // retrieve candidates for the current word
            Object[] featVector = FeatureExtractor.extractACFeatures(pIdx, wordIdx, sentence, numOfFeatures, indexMap);
            double[] labelScores = acClassifier.score(featVector);

            // build an intermediate beam
            TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

            for (int index = 0; index < currentBeam.size(); index++) {
                double currentScore = currentBeam.get(index).first;

                for (int labelIdx = 0; labelIdx < labelMap.length; labelIdx++) {
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


    //this function is used for joint ai-ac decoding
    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestCandidatesLiblinear
            (Sentence sentence, int pIdx, IndexMap indexMap,
             int maxBeamSize, int numOfFeatures,
             HashMap<Object, Integer>[] featDict) throws Exception {
        int[] labels = acClassifier_ll.getLabels();

        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));


        // Gradual building of the beam for all words in the sentence
        for (int wordIdx = 1; wordIdx < sentence.getWords().length; wordIdx++) {
            // retrieve candidates for the current word
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
            double[] labelScores = new double[labels.length];
            int prediction = (int) Linear.predictProbability(acClassifier_ll, featureNodes, probEstimates);
            for (int labelIdx=0; labelIdx< labels.length; labelIdx++)
                labelScores[labelIdx] = Math.exp(probEstimates[labelIdx]);

            // build an intermediate beam
            TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

            for (int index = 0; index < currentBeam.size(); index++) {
                double currentScore = currentBeam.get(index).first;

                for (int labelIdx = 0; labelIdx < labels.length; labelIdx++) {
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


    private ArrayList<Pair<Double, ArrayList<Integer>>> getBestCandidatesAdam
            (Sentence sentence, int pIdx, IndexMap indexMap,
             int maxBeamSize, int numOfFeatures,
             HashMap<Object, Integer>[] featDict) throws Exception {
        String[] labelMap = acClassifier_adam.getLabelMap();

        ArrayList<Pair<Double, ArrayList<Integer>>> currentBeam = new ArrayList<Pair<Double, ArrayList<Integer>>>();
        currentBeam.add(new Pair<Double, ArrayList<Integer>>(0., new ArrayList<Integer>()));


        // Gradual building of the beam for all words in the sentence
        for (int wordIdx = 1; wordIdx < sentence.getWords().length; wordIdx++) {
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
            double[] labelScores = new double[labelMap.length];
            int prediction = acClassifier_adam.argmax(feats, probEstimates);
            for (int labelIdx=0; labelIdx< labelMap.length; labelIdx++)
                labelScores[labelIdx] = Math.log(probEstimates[labelIdx]);

            // build an intermediate beam
            TreeSet<BeamElement> newBeamHeap = new TreeSet<BeamElement>();

            for (int index = 0; index < currentBeam.size(); index++) {
                double currentScore = currentBeam.get(index).first;

                for (int labelIdx = 0; labelIdx < labelMap.length; labelIdx++) {
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


    public HashMap<Integer, Prediction> predictAI(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                                                  int numOfFeatures, String modelDir, int numOfPDFeatures,
                                                  HashMap<Object, Integer>[] featDict,
                                                  ClassifierType classifierType)
            throws Exception {


        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);
        HashMap<Integer, Prediction> predictedPAs = new HashMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates = new ArrayList();

            if (classifierType == ClassifierType.AveragedPerceptron)
                aiCandidates= getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfFeatures);
            else if (classifierType == ClassifierType.Liblinear)
                aiCandidates= getBestAICandidatesLiblinear(sentence, pIdx, indexMap, aiMaxBeamSize, numOfFeatures, featDict);

            HashMap<Integer, Integer> highestScorePrediction = getHighestScorePredication(aiCandidates);
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }


    public HashMap<Integer, Prediction> predictAC(Sentence sentence, IndexMap indexMap,
                                                  int acMaxBeamSize, int aiMaxBeamSize, int numOfAIFeatures,
                                                  int numOfACFeatures, int numOfPDFeatures, String modelDir,
                                                  HashMap<Object, Integer>[] featDict,
                                                  ClassifierType classifierType) throws Exception {


        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);
        HashMap<Integer, Prediction> predictedPAs = new HashMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates= new ArrayList();
            ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates = new ArrayList();

            if (classifierType == ClassifierType.AveragedPerceptron) {
                aiCandidates = getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfAIFeatures);
                // get best <=l argument label for each of these k assignments
                acCandidates=getBestACCandidates(sentence, pIdx, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures);
            }else if (classifierType == ClassifierType.Liblinear)
            {
                aiCandidates = getBestAICandidatesLiblinear(sentence, pIdx, indexMap, aiMaxBeamSize, numOfAIFeatures, featDict);
                // get best <=l argument label for each of these k assignments
                acCandidates=getBestACCandidatesLiblinear(sentence, pIdx, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures, featDict);
            }

            HashMap<Integer, Integer> highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates);
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }



    public TreeMap<Integer, Prediction> predict(Sentence sentence, IndexMap indexMap, int aiMaxBeamSize,
                                                int acMaxBeamSize, int numOfAIFeatures, int numOfACFeatures,
                                                int numOfPDFeatures, String modelDir,
                                                HashMap<Object, Integer>[] aiFeatDict,
                                                HashMap<Object, Integer>[] acFeatDict,
                                                ClassifierType classifierType) throws Exception {


        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);
        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            // get best k argument assignment candidates
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates= new ArrayList();
            ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates= new ArrayList();
            if (classifierType == ClassifierType.AveragedPerceptron) {

                aiCandidates = getBestAICandidates(sentence, pIdx, indexMap, aiMaxBeamSize, numOfAIFeatures);
                acCandidates = getBestACCandidates(sentence, pIdx, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures);

            }else if (classifierType == ClassifierType.Liblinear)
            {
                aiCandidates = getBestAICandidatesLiblinear(sentence, pIdx, indexMap, aiMaxBeamSize, numOfAIFeatures, aiFeatDict);
                acCandidates = getBestACCandidatesLiblinear(sentence, pIdx, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures, acFeatDict);

            }else if (classifierType == ClassifierType.Adam)
            {
                aiCandidates = getBestAICandidatesAdam(sentence, pIdx, indexMap, aiMaxBeamSize, numOfAIFeatures, aiFeatDict);
                acCandidates = getBestACCandidatesAdam(sentence, pIdx, indexMap, aiCandidates, acMaxBeamSize, numOfACFeatures, acFeatDict);
            }

            HashMap<Integer, Integer> highestScorePrediction = getHighestScorePredication(aiCandidates, acCandidates);
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));
        }
        return predictedPAs;
    }

    //this function is used for joint ai-ac decoding
    public TreeMap<Integer, Prediction> predictJoint(Sentence sentence, IndexMap indexMap,
                                                     int maxBeamSize, int numOfFeatures, int numOfPDFeatures,
                                                     String modelDir, HashMap<Object, Integer>[] featDict,
                                                     ClassifierType classifierType) throws Exception {

        HashMap<Integer, String> predictedPredicates = PD.predict(sentence, indexMap, modelDir, numOfPDFeatures);
        TreeMap<Integer, Prediction> predictedPAs = new TreeMap<Integer, Prediction>();

        for (int pIdx : predictedPredicates.keySet()) {
            String pLabel = predictedPredicates.get(pIdx);
            ArrayList<Pair<Double, ArrayList<Integer>>> candidates= new ArrayList();
            if (classifierType == ClassifierType.AveragedPerceptron)
                   candidates = getBestCandidates(sentence, pIdx, indexMap, maxBeamSize, numOfFeatures);
            else if (classifierType == ClassifierType.Liblinear)
                candidates = getBestCandidatesLiblinear(sentence, pIdx, indexMap, maxBeamSize, numOfFeatures, featDict);
            else if (classifierType == ClassifierType.Adam)
                candidates = getBestCandidatesAdam(sentence, pIdx, indexMap, maxBeamSize, numOfFeatures, featDict);

            HashMap<Integer, Integer> highestScorePrediction = getHighestScorePredicationJoint(candidates, pIdx);
            predictedPAs.put(pIdx, new Prediction(pLabel, highestScorePrediction));

        }
        return predictedPAs;
    }

}
