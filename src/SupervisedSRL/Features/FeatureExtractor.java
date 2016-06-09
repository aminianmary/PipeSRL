package SupervisedSRL.Features;
/**
 * Created by Maryam Aminian on 5/17/16.
 */

import Sentence.Sentence;
import Sentence.Predicate;
import Sentence.Argument;
import apple.laf.JRSUIUtils;
import util.StringUtils;

import java.util.HashMap;
import java.util.TreeSet;
import java.util.Set;

public class FeatureExtractor {

    public static String[] extractFeatures(Predicate p, int aIdx, Sentence sentence, String state, int length) {
        String[] features = new String[length];
        String[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        String[] sentenceFeats = sentence.getFeats();
        String[] sentenceWords = sentence.getWords();
        String[] sentencePOSTags = sentence.getPosTags();
        String[] sentenceLemmas = sentence.getLemmas();
        HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads = sentence.getReverseDepHeads();

        //predicate features
        int pIdx = p.getIndex();
        String pw = sentenceWords[pIdx];
        String ppos = sentencePOSTags[pIdx];
        String plem = sentenceLemmas[pIdx];
        String pdeprel = sentenceDepLabels[pIdx];
        String pfeats = sentenceFeats[pIdx];
        String psense = p.getLabel();
        String pprw = sentenceWords[sentenceDepHeads[pIdx]];
        String pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
        String pprfeats = sentenceFeats[sentenceDepHeads[pIdx]];
        String pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels);
        String pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels);
        String pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags);
        String pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords);

        String voice = sentence.getVoice(pIdx);

        //role label
        // String label= a.getType();
        if (state.equals("AI") || state.equals("AC")) {
            int leftMostDependentIndex = getLeftMostDependentIndex(aIdx, sentenceReverseDepHeads);
            int rightMostDependentIndex = getRightMostDependentIndex(aIdx, sentenceReverseDepHeads);
            int lefSiblingIndex = getLeftSiblingIndex(aIdx, sentenceReverseDepHeads);
            int rightSiblingIndex = getRightSiblingIndex(aIdx, sentenceReverseDepHeads);

            //argument features
            String aw = sentenceWords[aIdx];
            String apos = sentencePOSTags[aIdx];
            String afeat = sentenceFeats[aIdx];
            String adeprel = sentenceDepLabels[aIdx];

            //predicate-argument features
            String deprelpath = StringUtils.convertPathArrayIntoString(sentence.getDepPath(pIdx, aIdx));
            String pospath = StringUtils.convertPathArrayIntoString(sentence.getPOSPath(pIdx, aIdx));
            String position = (pIdx < aIdx) ? "a" : "b";
            String leftw = (leftMostDependentIndex != -1) ? sentenceWords[leftMostDependentIndex] : "";
            String leftpos = (leftMostDependentIndex != -1) ? sentencePOSTags[leftMostDependentIndex] : "";
            String leftfeats = (leftMostDependentIndex != -1) ? sentenceFeats[leftMostDependentIndex] : "";
            String rightw = (rightMostDependentIndex != -1) ? sentenceWords[rightMostDependentIndex] : "";
            String rightpos = (rightMostDependentIndex != -1) ? sentencePOSTags[rightMostDependentIndex] : "";
            String rightfeats = (rightMostDependentIndex != -1) ? sentenceFeats[rightMostDependentIndex] : "";
            String rightsiblingw = (rightSiblingIndex != -1) ? sentenceWords[rightSiblingIndex] : "";
            String rightsiblingpos = (rightSiblingIndex != -1) ? sentencePOSTags[rightSiblingIndex] : "";
            String rightsiblingfeats = (rightSiblingIndex != -1) ? sentenceFeats[rightSiblingIndex] : "";
            String leftsiblingw = (lefSiblingIndex != -1) ? sentenceWords[lefSiblingIndex] : "";
            String leftsiblingpos = (lefSiblingIndex != -1) ? sentencePOSTags[lefSiblingIndex] : "";
            String leftsiblingfeats = (lefSiblingIndex != -1) ? sentenceFeats[lefSiblingIndex] : "";


            //build feature vector for argument identification module
            //first order features
            int index = 0;
            features[index++] = pw;
            features[index++] = ppos;
            features[index++] = plem;
            features[index++] = pdeprel;
            features[index++] = psense;
            // todo features should be separated
            features[index++] = pfeats;
            features[index++] = pprw;
            features[index++] = pprpos;
            features[index++] = pprfeats;
            features[index++] = pdepsubcat;
            features[index++] = pchilddepset;
            features[index++] = pchildposset;
            features[index++] = pchildwset;

            features[index++] = aw;
            features[index++] = apos;
            features[index++] = afeat;
            features[index++] = adeprel;
            features[index++] = deprelpath;
            features[index++] = pospath;
            features[index++] = position;
            features[index++] = leftw;
            features[index++] = leftpos;
            features[index++] = leftfeats;
            features[index++] = rightw;
            features[index++] = rightpos;
            features[index++] = rightfeats;
            features[index++] = leftsiblingw;
            features[index++] = leftsiblingpos;
            features[index++] = leftsiblingfeats;
            features[index++] = rightsiblingw;
            features[index++] = rightsiblingpos;
            features[index++] = rightsiblingfeats;

            //predicate-argument conjoined features
            features[index++] = pw + "_" + aw;
            features[index++] = pw + "_" + apos;
            features[index++] = pw + "_" + afeat;
            features[index++] = pw + "_" + adeprel;
            features[index++] = pw + "_" + deprelpath;
            features[index++] = pw + "_" + pospath;
            features[index++] = pw + "_" + position;
            features[index++] = pw + "_" + leftw;
            features[index++] = pw + "_" + leftpos;
            features[index++] = pw + "_" + leftfeats;
            features[index++] = pw + "_" + rightw;
            features[index++] = pw + "_" + rightpos;
            features[index++] = pw + "_" + rightfeats;
            features[index++] = pw + "_" + leftsiblingw;
            features[index++] = pw + "_" + leftsiblingpos;
            features[index++] = pw + "_" + leftsiblingfeats;
            features[index++] = pw + "_" + rightsiblingw;
            features[index++] = pw + "_" + rightsiblingpos;
            features[index++] = pw + "_" + rightsiblingfeats;

            features[index++] = ppos + "_" + aw;
            features[index++] = ppos + "_" + apos;
            features[index++] = ppos + "_" + afeat;
            features[index++] = ppos + "_" + adeprel;
            features[index++] = ppos + "_" + deprelpath;
            features[index++] = ppos + "_" + pospath;
            features[index++] = ppos + "_" + position;
            features[index++] = ppos + "_" + leftw;
            features[index++] = ppos + "_" + leftpos;
            features[index++] = ppos + "_" + leftfeats;
            features[index++] = ppos + "_" + rightw;
            features[index++] = ppos + "_" + rightpos;
            features[index++] = ppos + "_" + rightfeats;
            features[index++] = ppos + "_" + leftsiblingw;
            features[index++] = ppos + "_" + leftsiblingpos;
            features[index++] = ppos + "_" + leftsiblingfeats;
            features[index++] = ppos + "_" + rightsiblingw;
            features[index++] = ppos + "_" + rightsiblingpos;
            features[index++] = ppos + "_" + rightsiblingfeats;


            features[index++] = pdeprel + "_" + aw;
            features[index++] = pdeprel + "_" + apos;
            features[index++] = pdeprel + "_" + afeat;
            features[index++] = pdeprel + "_" + adeprel;
            features[index++] = pdeprel + "_" + deprelpath;
            features[index++] = pdeprel + "_" + pospath;
            features[index++] = pdeprel + "_" + position;
            features[index++] = pdeprel + "_" + leftw;
            features[index++] = pdeprel + "_" + leftpos;
            features[index++] = pdeprel + "_" + leftfeats;
            features[index++] = pdeprel + "_" + rightw;
            features[index++] = pdeprel + "_" + rightpos;
            features[index++] = pdeprel + "_" + rightfeats;
            features[index++] = pdeprel + "_" + leftsiblingw;
            features[index++] = pdeprel + "_" + leftsiblingpos;
            features[index++] = pdeprel + "_" + leftsiblingfeats;
            features[index++] = pdeprel + "_" + rightsiblingw;
            features[index++] = pdeprel + "_" + rightsiblingpos;
            features[index++] = pdeprel + "_" + rightsiblingfeats;

            features[index++] = pdepsubcat + "_" + aw;
            features[index++] = pdepsubcat + "_" + apos;
            features[index++] = pdepsubcat + "_" + adeprel;
            features[index++] = pdepsubcat + "_" + position;
        }

        //build feature vector for predicate disambiguation module
        if (state.equals("PD")) {
            int index = 0;
            features[index++] = pw;
            features[index++] = ppos;
            features[index++] = pdeprel;
            features[index++] = pfeats;
            features[index++] = pprw;
            features[index++] = pprpos;
            features[index++] = pprfeats;
            features[index++] = pchilddepset;
            features[index++] = pchildposset;
            features[index++] = pchildwset;
        }

        return features;
    }

    //TODO dependency subcat frames should contain core dep labels (not all of them)
    public static String getDepSubCat(int pIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads,
                                      String[] sentenceDepLabels) {
        String subCat = "";
        if (sentenceReverseDepHeads.containsKey(pIdx) && sentenceReverseDepHeads.get(pIdx).size() > 0) {
            for (int child : sentenceReverseDepHeads.get(pIdx))
                subCat += sentenceDepLabels[child] + "\t";
        }
        return subCat.trim().replaceAll("\t", "_");
    }

    public static String getChildSet(int pIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads,
                                     String[] collection) {
        String subCat = "";
        if (sentenceReverseDepHeads.containsKey(pIdx) && sentenceReverseDepHeads.get(pIdx).size() > 0) {
            for (int child : sentenceReverseDepHeads.get(pIdx))
                subCat += collection[child] + "\t";
        }
        return subCat.trim().replaceAll("\t", "|");
    }

    public static int getLeftMostDependentIndex(int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size() > 0)
            return sentenceReverseDepHeads.get(aIdx).last();
        return -1;
    }

    public static int getRightMostDependentIndex(int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size() > 0)
            return sentenceReverseDepHeads.get(aIdx).first();
        return -1;
    }

    public static int getLeftSiblingIndex(int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size() > 0
                && sentenceReverseDepHeads.get(aIdx).higher(aIdx) != null)
            return sentenceReverseDepHeads.get(aIdx).higher(aIdx);
        return -1;
    }

    public static int getRightSiblingIndex(int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size() > 0
                && sentenceReverseDepHeads.get(aIdx).lower(aIdx) != null)
            return sentenceReverseDepHeads.get(aIdx).lower(aIdx);
        return -1;
    }


}
