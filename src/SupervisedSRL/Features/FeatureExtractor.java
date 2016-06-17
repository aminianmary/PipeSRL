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
        String pprfeats = (sentenceFeats[sentenceDepHeads[pIdx]]==null)?"":sentenceFeats[sentenceDepHeads[pIdx]];
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
            for (String feat:pfeats.split("|"))
                features[index++] = feat;
            features[index++] = pprw;
            features[index++] = pprpos;
            for (String feat:pprfeats.split("|"))
                features[index++] = feat;
            features[index++] = pdepsubcat;
            features[index++] = pchilddepset;
            features[index++] = pchildposset;
            features[index++] = pchildwset;

            features[index++] = aw;
            features[index++] = apos;
            for (String feat:afeat.split("|"))
                features[index++] = feat;
            features[index++] = adeprel;
            features[index++] = deprelpath;
            features[index++] = pospath;
            features[index++] = position;
            features[index++] = leftw;
            features[index++] = leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = feat;
            features[index++] = rightw;
            features[index++] = rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = feat;
            features[index++] = leftsiblingw;
            features[index++] = leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = feat;
            features[index++] = rightsiblingw;
            features[index++] = rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = feat;

            //predicate-argument conjoined features
            // pw + argument features
            features[index++] = pw + "_" + aw;
            features[index++] = pw + "_" + apos;
            for (String feat:afeat.split("|"))
                features[index++] = pw + "_" + feat;
            features[index++] = pw + "_" + adeprel;
            features[index++] = pw + "_" + deprelpath;
            features[index++] = pw + "_" + pospath;
            features[index++] = pw + "_" + position;
            features[index++] = pw + "_" + leftw;
            features[index++] = pw + "_" + leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = pw + "_" + feat;
            features[index++] = pw + "_" + rightw;
            features[index++] = pw + "_" + rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = pw + "_" + feat;
            features[index++] = pw + "_" + leftsiblingw;
            features[index++] = pw + "_" + leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = pw + "_" + feat;
            features[index++] = pw + "_" + rightsiblingw;
            features[index++] = pw + "_" + rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = pw + "_" + feat;

            //ppos + argument features
            features[index++] = ppos + "_" + aw;
            features[index++] = ppos + "_" + apos;
            for (String feat:afeat.split("|"))
                features[index++] = ppos + "_" + feat;
            features[index++] = ppos + "_" + adeprel;
            features[index++] = ppos + "_" + deprelpath;
            features[index++] = ppos + "_" + pospath;
            features[index++] = ppos + "_" + position;
            features[index++] = ppos + "_" + leftw;
            features[index++] = ppos + "_" + leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = ppos + "_" + feat;
            features[index++] = ppos + "_" + rightw;
            features[index++] = ppos + "_" + rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = ppos + "_" + feat;
            features[index++] = ppos + "_" + leftsiblingw;
            features[index++] = ppos + "_" + leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = ppos + "_" + feat;
            features[index++] = ppos + "_" + rightsiblingw;
            features[index++] = ppos + "_" + rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = ppos + "_" + feat;

            //pdeprel + argument features
            features[index++] = pdeprel + "_" + aw;
            features[index++] = pdeprel + "_" + apos;
            for (String feat:afeat.split("|"))
                features[index++] = pdeprel + "_" + feat;
            features[index++] = pdeprel + "_" + adeprel;
            features[index++] = pdeprel + "_" + deprelpath;
            features[index++] = pdeprel + "_" + pospath;
            features[index++] = pdeprel + "_" + position;
            features[index++] = pdeprel + "_" + leftw;
            features[index++] = pdeprel + "_" + leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = pdeprel + "_" + feat;
            features[index++] = pdeprel + "_" + rightw;
            features[index++] = pdeprel + "_" + rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = pdeprel + "_" + feat;
            features[index++] = pdeprel + "_" + leftsiblingw;
            features[index++] = pdeprel + "_" + leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = pdeprel + "_" + feat;
            features[index++] = pdeprel + "_" + rightsiblingw;
            features[index++] = pdeprel + "_" + rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = pdeprel + "_" + feat;


            //plem + argument features
            features[index++] = plem + "_" + aw;
            features[index++] = plem + "_" + apos;
            for (String feat:afeat.split("|"))
                features[index++] = plem + "_" + feat;
            features[index++] = plem + "_" + adeprel;
            features[index++] = plem + "_" + deprelpath;
            features[index++] = plem + "_" + pospath;
            features[index++] = plem + "_" + position;
            features[index++] = plem + "_" + leftw;
            features[index++] = plem + "_" + leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = plem + "_" + feat;
            features[index++] = plem + "_" + rightw;
            features[index++] = plem + "_" + rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = plem + "_" + feat;
            features[index++] = plem + "_" + leftsiblingw;
            features[index++] = plem + "_" + leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = plem + "_" + feat;
            features[index++] = plem + "_" + rightsiblingw;
            features[index++] = plem + "_" + rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = plem + "_" + feat;


            //psense + argument features
            features[index++] = psense + "_" + aw;
            features[index++] = psense + "_" + apos;
            for (String feat:afeat.split("|"))
                features[index++] = psense + "_" + feat;
            features[index++] = psense + "_" + adeprel;
            features[index++] = psense + "_" + deprelpath;
            features[index++] = psense + "_" + pospath;
            features[index++] = psense + "_" + position;
            features[index++] = psense + "_" + leftw;
            features[index++] = psense + "_" + leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = psense + "_" + feat;
            features[index++] = psense + "_" + rightw;
            features[index++] = psense + "_" + rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = psense + "_" + feat;
            features[index++] = psense + "_" + leftsiblingw;
            features[index++] = psense + "_" + leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = psense + "_" + feat;
            features[index++] = psense + "_" + rightsiblingw;
            features[index++] = psense + "_" + rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = psense + "_" + feat;


            //pprw + argument features
            features[index++] = pprw + "_" + aw;
            features[index++] = pprw + "_" + apos;
            for (String feat:afeat.split("|"))
                features[index++] = pprw + "_" + feat;
            features[index++] = pprw + "_" + adeprel;
            features[index++] = pprw + "_" + deprelpath;
            features[index++] = pprw + "_" + pospath;
            features[index++] = pprw + "_" + position;
            features[index++] = pprw + "_" + leftw;
            features[index++] = pprw + "_" + leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = pprw + "_" + feat;
            features[index++] = pprw + "_" + rightw;
            features[index++] = pprw + "_" + rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = pprw + "_" + feat;
            features[index++] = pprw + "_" + leftsiblingw;
            features[index++] = pprw + "_" + leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = pprw + "_" + feat;
            features[index++] = pprw + "_" + rightsiblingw;
            features[index++] = pprw + "_" + rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = pprw + "_" + feat;


            //pprpos + argument features
            features[index++] = pprpos + "_" + aw;
            features[index++] = pprpos + "_" + apos;
            for (String feat:afeat.split("|"))
                features[index++] = pprpos + "_" + feat;
            features[index++] = pprpos + "_" + adeprel;
            features[index++] = pprpos + "_" + deprelpath;
            features[index++] = pprpos + "_" + pospath;
            features[index++] = pprpos + "_" + position;
            features[index++] = pprpos + "_" + leftw;
            features[index++] = pprpos + "_" + leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = pprpos + "_" + feat;
            features[index++] = pprpos + "_" + rightw;
            features[index++] = pprpos + "_" + rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = pprpos + "_" + feat;
            features[index++] = pprpos + "_" + leftsiblingw;
            features[index++] = pprpos + "_" + leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = pprpos + "_" + feat;
            features[index++] = pprpos + "_" + rightsiblingw;
            features[index++] = pprpos + "_" + rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = pprpos + "_" + feat;


            //pchilddepset + argument features
            features[index++] = pchilddepset + "_" + aw;
            features[index++] = pchilddepset + "_" + apos;
            for (String feat:afeat.split("|"))
                features[index++] = pchilddepset + "_" + feat;
            features[index++] = pchilddepset + "_" + adeprel;
            features[index++] = pchilddepset + "_" + deprelpath;
            features[index++] = pchilddepset + "_" + pospath;
            features[index++] = pchilddepset + "_" + position;
            features[index++] = pchilddepset + "_" + leftw;
            features[index++] = pchilddepset + "_" + leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = pchilddepset + "_" + feat;
            features[index++] = pchilddepset + "_" + rightw;
            features[index++] = pchilddepset + "_" + rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = pchilddepset + "_" + feat;
            features[index++] = pchilddepset + "_" + leftsiblingw;
            features[index++] = pchilddepset + "_" + leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = pchilddepset + "_" + feat;
            features[index++] = pchilddepset + "_" + rightsiblingw;
            features[index++] = pchilddepset + "_" + rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = pchilddepset + "_" + feat;


            //pdepsubcat + argument features
            features[index++] = pdepsubcat + "_" + aw;
            features[index++] = pdepsubcat + "_" + apos;
            for (String feat:afeat.split("|"))
                features[index++] = pdepsubcat + "_" + feat;
            features[index++] = pdepsubcat + "_" + adeprel;
            features[index++] = pdepsubcat + "_" + deprelpath;
            features[index++] = pdepsubcat + "_" + pospath;
            features[index++] = pdepsubcat + "_" + position;
            features[index++] = pdepsubcat + "_" + leftw;
            features[index++] = pdepsubcat + "_" + leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = pdepsubcat + "_" + feat;
            features[index++] = pdepsubcat + "_" + rightw;
            features[index++] = pdepsubcat + "_" + rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = pdepsubcat + "_" + feat;
            features[index++] = pdepsubcat + "_" + leftsiblingw;
            features[index++] = pdepsubcat + "_" + leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = pdepsubcat + "_" + feat;
            features[index++] = pdepsubcat + "_" + rightsiblingw;
            features[index++] = pdepsubcat + "_" + rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = pdepsubcat + "_" + feat;

            //pchilddepset + argument features
            features[index++] = pchilddepset + "_" + aw;
            features[index++] = pchilddepset + "_" + apos;
            for (String feat:afeat.split("|"))
                features[index++] = pchilddepset + "_" + feat;
            features[index++] = pchilddepset + "_" + adeprel;
            features[index++] = pchilddepset + "_" + deprelpath;
            features[index++] = pchilddepset + "_" + pospath;
            features[index++] = pchilddepset + "_" + position;
            features[index++] = pchilddepset + "_" + leftw;
            features[index++] = pchilddepset + "_" + leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = pchilddepset + "_" + feat;
            features[index++] = pchilddepset + "_" + rightw;
            features[index++] = pchilddepset + "_" + rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = pchilddepset + "_" + feat;
            features[index++] = pchilddepset + "_" + leftsiblingw;
            features[index++] = pchilddepset + "_" + leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = pchilddepset + "_" + feat;
            features[index++] = pchilddepset + "_" + rightsiblingw;
            features[index++] = pchilddepset + "_" + rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = pchilddepset + "_" + feat;

            //pchildwset + argument features
            features[index++] = pchildwset + "_" + aw;
            features[index++] = pchildwset + "_" + apos;
            for (String feat:afeat.split("|"))
                features[index++] = pchildwset + "_" + feat;
            features[index++] = pchildwset + "_" + adeprel;
            features[index++] = pchildwset + "_" + deprelpath;
            features[index++] = pchildwset + "_" + pospath;
            features[index++] = pchildwset + "_" + position;
            features[index++] = pchildwset + "_" + leftw;
            features[index++] = pchildwset + "_" + leftpos;
            for (String feat:leftfeats.split("|"))
                features[index++] = pchildwset + "_" + feat;
            features[index++] = pchildwset + "_" + rightw;
            features[index++] = pchildwset + "_" + rightpos;
            for (String feat:rightfeats.split("|"))
                features[index++] = pchildwset + "_" + feat;
            features[index++] = pchildwset + "_" + leftsiblingw;
            features[index++] = pchildwset + "_" + leftsiblingpos;
            for (String feat:leftsiblingfeats.split("|"))
                features[index++] = pchildwset + "_" + feat;
            features[index++] = pchildwset + "_" + rightsiblingw;
            features[index++] = pchildwset + "_" + rightsiblingpos;
            for (String feat:rightsiblingfeats.split("|"))
                features[index++] = pchildwset + "_" + feat;

            //pw + aw + apos
            features[index++] = plem + aw + apos;
            features[index++] = plem + aw + adeprel;
            features[index++] = ppos + apos + adeprel;
            features[index++] = pdeprel + apos + adeprel;
            features[index++] = pchilddepset + apos + adeprel;
            features[index++] = pchilddepset + apos + adeprel;
            features[index++] = pchildwset + aw + adeprel;
            features[index++] = plem + apos + adeprel;
        }

        //build feature vector for predicate disambiguation module
        if (state.equals("PD")) {
            int index = 0;
            features[index++] = pw;
            features[index++] = ppos;
            features[index++] = pdeprel;
            for (String feat:pfeats.split("|"))
                features[index++] = feat;
            features[index++] = pprw;
            features[index++] = pprpos;
            for (String feat:pprfeats.split("|"))
                features[index++] = feat;
            features[index++] = pchilddepset;
            features[index++] = pchildposset;
            features[index++] = pchildwset;
        }

        return features;
    }

    //TODO dependency subcat frames should contain core dep labels (not all of them)
    private static String getDepSubCat(int pIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads,
                                      String[] sentenceDepLabels) {
        String subCat = "";
        TreeSet<String> subCatElements= new TreeSet<String>();
        if (sentenceReverseDepHeads.containsKey(pIdx) && sentenceReverseDepHeads.get(pIdx).size() > 0) {
            for (int child : sentenceReverseDepHeads.get(pIdx))
                subCatElements.add(sentenceDepLabels[child]);
        }

        for (String str: subCatElements)
            subCat +=  str + "\t";
        return subCat.trim().replaceAll("\t", "+");
    }

    private static String getChildSet(int pIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads,
                                     String[] collection) {
        String subCat = "";
        TreeSet<String> childs= new TreeSet<String>();
        if (sentenceReverseDepHeads.containsKey(pIdx) && sentenceReverseDepHeads.get(pIdx).size() > 0) {
            for (int child : sentenceReverseDepHeads.get(pIdx))
                childs.add(collection[child]);
        }
        for (String str: childs)
         subCat += str+"\t";
        return subCat.trim().replaceAll("\t", "|");
    }

    private static int getLeftMostDependentIndex(int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size() > 0)
            return sentenceReverseDepHeads.get(aIdx).last();
        return -1;
    }

    private static int getRightMostDependentIndex(int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size() > 0)
            return sentenceReverseDepHeads.get(aIdx).first();
        return -1;
    }

    private static int getLeftSiblingIndex(int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size() > 0
                && sentenceReverseDepHeads.get(aIdx).higher(aIdx) != null)
            return sentenceReverseDepHeads.get(aIdx).higher(aIdx);
        return -1;
    }

    private static int getRightSiblingIndex(int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size() > 0
                && sentenceReverseDepHeads.get(aIdx).lower(aIdx) != null)
            return sentenceReverseDepHeads.get(aIdx).lower(aIdx);
        return -1;
    }

}
