package SupervisedSRL.Features;
/**
 * Created by Maryam Aminian on 5/17/16.
 */

import Sentence.Sentence;
import SupervisedSRL.Strcutures.IndexMap;
import util.StringUtils;

import java.util.HashSet;
import java.util.TreeSet;

public class FeatureExtractor {
    static HashSet<String> punctuations = new HashSet<String>();

    static {
        punctuations.add("P");
        punctuations.add("PUNC");
        punctuations.add("PUNCT");
        punctuations.add("p");
        punctuations.add("punc");
        punctuations.add("punct");
        punctuations.add(",");
        punctuations.add(";");
        punctuations.add(".");
        punctuations.add("#");
        punctuations.add("''");
        punctuations.add("(");
        punctuations.add(")");
        punctuations.add("[");
        punctuations.add("]");
        punctuations.add("{");
        punctuations.add("}");
        punctuations.add("\"");
        punctuations.add(",");
        punctuations.add(".");
        punctuations.add(":");
        punctuations.add("``");
        punctuations.add("-LRB-");
        punctuations.add("-RRB-");
        punctuations.add("-LSB-");
        punctuations.add("-RSB-");
        punctuations.add("-LCB-");
        punctuations.add("-RCB-");
        punctuations.add("!");
        punctuations.add(".");
        punctuations.add("#");
        punctuations.add("$");
        punctuations.add("''");
        punctuations.add("(");
        punctuations.add(")");
        punctuations.add(",");
        punctuations.add("-LRB-");
        punctuations.add("-RRB-");
        punctuations.add(":");
        punctuations.add("?");
    }

    public static Object[] extractPDFeatures(int pIdx, String pSense, int aIdx, Sentence sentence, int length,
                                             IndexMap indexMap) throws Exception {
        Object[] features = new Object[length];
        int[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        int[] sentenceWords = sentence.getWords();
        int[] sentencePOSTags = sentence.getPosTags();
        TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();

        //predicate features
        int pw = sentenceWords[pIdx];
        int ppos = sentencePOSTags[pIdx];
        int pdeprel = sentenceDepLabels[pIdx];
        int pprw = sentenceWords[sentenceDepHeads[pIdx]];
        int pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
        String pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags, indexMap);
        String pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags, indexMap);

        int index = 0;
        features[index++] = pw;
        features[index++] = ppos;
        features[index++] = pdeprel;
        features[index++] = pprw;
        features[index++] = pprpos;
        features[index++] = pdepsubcat;
        features[index++] = pchilddepset;
        features[index++] = pchildposset;
        features[index++] = pchildwset;

        return features;
    }

    public static Object[] extractAIFeatures(int pIdx, String pSense, int aIdx, Sentence sentence, int length,
                                             IndexMap indexMap) throws Exception {
        // todo object; int; respectively
        Object[] features = new Object[length];
        int[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        int[] sentenceWords = sentence.getWords();
        int[] sentencePOSTags = sentence.getPosTags();
        int[] sentenceLemmas = sentence.getLemmas();
        TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();

        //predicate features
        int pw = sentenceWords[pIdx];
        int ppos = sentencePOSTags[pIdx];
        int plem = sentenceLemmas[pIdx];
        int pdeprel = sentenceDepLabels[pIdx];
        int pprw = sentenceWords[sentenceDepHeads[pIdx]];
        int pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
        String pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags, indexMap);
        String pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags, indexMap);

        int leftMostDependentIndex = getLeftMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int rightMostDependentIndex = getRightMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int lefSiblingIndex = getLeftSiblingIndex(aIdx, pIdx, sentenceReverseDepHeads);
        int rightSiblingIndex = getRightSiblingIndex(aIdx, pIdx, sentenceReverseDepHeads);

        //argument features
        int aw = sentenceWords[aIdx];
        int apos = sentencePOSTags[aIdx];
        int adeprel = sentenceDepLabels[aIdx];

        //predicate-argument features
        String deprelpath = StringUtils.convertPathArrayIntoString(sentence.getDepPath(pIdx, aIdx));
        String pospath = StringUtils.convertPathArrayIntoString(sentence.getPOSPath(pIdx, aIdx));

        int position = 0; //on
        if (pIdx < aIdx)
            position = 1; //before
        else
            position = 2; //after

        int leftw = (leftMostDependentIndex != -1) ? sentenceWords[leftMostDependentIndex] : indexMap.getNullIdx();
        int leftpos = (leftMostDependentIndex != -1) ? sentencePOSTags[leftMostDependentIndex] : indexMap.getNullIdx();
        int rightw = (rightMostDependentIndex != -1) ? sentenceWords[rightMostDependentIndex] : indexMap.getNullIdx();
        int rightpos = (rightMostDependentIndex != -1) ? sentencePOSTags[rightMostDependentIndex] : indexMap.getNullIdx();
        int rightsiblingw = (rightSiblingIndex != -1) ? sentenceWords[rightSiblingIndex] : indexMap.getNullIdx();
        int rightsiblingpos = (rightSiblingIndex != -1) ? sentencePOSTags[rightSiblingIndex] : indexMap.getNullIdx();
        int leftsiblingw = (lefSiblingIndex != -1) ? sentenceWords[lefSiblingIndex] : indexMap.getNullIdx();
        int leftsiblingpos = (lefSiblingIndex != -1) ? sentencePOSTags[lefSiblingIndex] : indexMap.getNullIdx();

        //build feature vector for argument identification/classification modules
        int index = 0;

        //////////////////////////////////////////////////////////////////////
        ////////////////////////// SINGLE FEATURES //////////////////////////
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////
        ////// PREDICATE FEATURES //////
        ////////////////////////////////
        features[index++] = pw;
        features[index++] = ppos;
        features[index++] = plem;
        features[index++] = pdeprel;
        features[index++] = pSense;
        features[index++] = pprw;
        features[index++] = pprpos;
        features[index++] = pdepsubcat;
        features[index++] = pchilddepset;
        features[index++] = pchildposset;
        features[index++] = pchildwset;

        ////////////////////////////////
        ////// ARGUMENT FEATURES //////
        ////////////////////////////////
        features[index++] = aw;
        features[index++] = apos;
        features[index++] = adeprel;
        features[index++] = deprelpath;
        features[index++] = pospath;
        features[index++] = position;
        features[index++] = leftw;
        features[index++] = leftpos;
        features[index++] = rightw;
        features[index++] = rightpos;
        features[index++] = leftsiblingw;
        features[index++] = leftsiblingpos;
        features[index++] = rightsiblingw;
        features[index++] = rightsiblingpos;

        //////////////////////////////////////////////////////////////////////
        ///////////////// CONJOINED FEATURES /////////////////////////////////
        //////////////////////////////////////////////////////////////////////

        //conjoined features added based on the original paper
        int aw_position = (aw << 2) | position;
        features[index++] = aw_position;
        String psense_position = pSense + " " + position;
        features[index++] = psense_position;
        String psense_apos = pSense + " " + apos;
        features[index++] = psense_apos;
        String deprelpath_position = deprelpath + " " + position;
        features[index++] = deprelpath_position;
        int pprw_adeprel = (pprw << 10) | adeprel;
        features[index++] = pprw_adeprel;
        int pw_leftpos = (pw << 10) | leftpos;
        features[index++] = pw_leftpos;
        String psense_pchildwset = pSense + " " + pchildwset;
        features[index++] = psense_pchildwset;
        String adeprel_deprelpath = adeprel + " " + deprelpath;
        features[index++] = adeprel_deprelpath;
        String pospath_rightsiblingpos = pospath + " " + rightsiblingpos;
        features[index++] = pospath_rightsiblingpos;
        String pchilddepset_adeprel = pchilddepset + " " + adeprel;
        features[index++] = pchilddepset_adeprel;
        int apos_adeprel = (apos << 10) | adeprel;
        features[index++] = apos_adeprel;
        int leftpos_rightpos = (leftpos << 10) | rightpos;
        features[index++] = leftpos_rightpos;
        int pdeprel_adeprel = (pdeprel << 10) | adeprel;
        features[index++] = pdeprel_adeprel;

        return features;
    }

    public static Object[] extractACFeatures(int pIdx, String pSense, int aIdx, Sentence sentence, int length,
                                             IndexMap indexMap) throws Exception {

        // todo object; int; respectively
        Object[] features = new Object[length];
        int[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        int[] sentenceWords = sentence.getWords();
        int[] sentencePOSTags = sentence.getPosTags();
        int[] sentenceLemmas = sentence.getLemmas();
        TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();

        //predicate features
        int pw = sentenceWords[pIdx];
        int ppos = sentencePOSTags[pIdx];
        int plem = sentenceLemmas[pIdx];
        int pdeprel = sentenceDepLabels[pIdx];
        int pprw = sentenceWords[sentenceDepHeads[pIdx]];
        int pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
        String pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags, indexMap);
        String pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags, indexMap);


        int leftMostDependentIndex = getLeftMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int rightMostDependentIndex = getRightMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int lefSiblingIndex = getLeftSiblingIndex(aIdx, pIdx, sentenceReverseDepHeads);
        int rightSiblingIndex = getRightSiblingIndex(aIdx, pIdx, sentenceReverseDepHeads);

        //argument features
        int aw = sentenceWords[aIdx];
        int apos = sentencePOSTags[aIdx];
        int adeprel = sentenceDepLabels[aIdx];

        //predicate-argument features
        String deprelpath = StringUtils.convertPathArrayIntoString(sentence.getDepPath(pIdx, aIdx));
        String pospath = StringUtils.convertPathArrayIntoString(sentence.getPOSPath(pIdx, aIdx));

        int position = 0; //on
        if (pIdx < aIdx)
            position = 1; //before
        else
            position = 2; //after

        int leftw = (leftMostDependentIndex != -1) ? sentenceWords[leftMostDependentIndex] : indexMap.getNullIdx();
        int leftpos = (leftMostDependentIndex != -1) ? sentencePOSTags[leftMostDependentIndex] : indexMap.getNullIdx();
        int rightw = (rightMostDependentIndex != -1) ? sentenceWords[rightMostDependentIndex] : indexMap.getNullIdx();
        int rightpos = (rightMostDependentIndex != -1) ? sentencePOSTags[rightMostDependentIndex] : indexMap.getNullIdx();
        int rightsiblingw = (rightSiblingIndex != -1) ? sentenceWords[rightSiblingIndex] : indexMap.getNullIdx();
        int rightsiblingpos = (rightSiblingIndex != -1) ? sentencePOSTags[rightSiblingIndex] : indexMap.getNullIdx();
        int leftsiblingw = (lefSiblingIndex != -1) ? sentenceWords[lefSiblingIndex] : indexMap.getNullIdx();
        int leftsiblingpos = (lefSiblingIndex != -1) ? sentencePOSTags[lefSiblingIndex] : indexMap.getNullIdx();

        //build feature vector for argument identification/classification modules
        int index = 0;

        //////////////////////////////////////////////////////////////////////
        ////////////////////////// SINGLE FEATURES //////////////////////////
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////
        ////// PREDICATE FEATURES //////
        ////////////////////////////////
        features[index++] = pw;
        features[index++] = ppos;
        features[index++] = plem;
        features[index++] = pdeprel;
        features[index++] = pSense;
        features[index++] = pprw;
        features[index++] = pprpos;
        features[index++] = pdepsubcat;
        features[index++] = pchilddepset;
        features[index++] = pchildposset;
        features[index++] = pchildwset;

        ////////////////////////////////
        ////// ARGUMENT FEATURES //////
        ////////////////////////////////
        features[index++] = aw;
        features[index++] = apos;
        features[index++] = adeprel;
        features[index++] = deprelpath;
        features[index++] = pospath;
        features[index++] = position;
        features[index++] = leftw;
        features[index++] = leftpos;
        features[index++] = rightw;
        features[index++] = rightpos;
        features[index++] = leftsiblingw;
        features[index++] = leftsiblingpos;
        features[index++] = rightsiblingw;
        features[index++] = rightsiblingpos;
        //////////////////////////////////////////////////////////////////////
        ///////////////// PREDICATE-ARGUMENT CONJOINED FEATURES (154) ////////
        //////////////////////////////////////////////////////////////////////

        // todo 20 bits for words, 10 bits for pos, 10 bits for dependencies
        // todo e.g. (pw<<20) | aw ==> 20+20> 32 ==> long
        // pw + argument features
        long pw_aw = (pw << 20) | aw;
        features[index++] = pw_aw;
        // todo (pw<<10) | pos ==> 10+20<32 ==> int
        int pw_apos = (pw << 10) | apos;
        features[index++] = pw_apos;
        int pw_adeprel = (pw << 10) | adeprel;
        features[index++] = pw_adeprel;
        String pw_deprelpath = pw + " " + deprelpath;
        features[index++] = pw_deprelpath;
        String pw_pospath = pw + " " + pospath;
        features[index++] = pw_pospath;
        int pw_position = (pw << 2) | position;
        features[index++] = pw_position;
        long pw_leftw = (pw << 20) | leftw;
        features[index++] = pw_leftw;
        int pw_leftpos = (pw << 10) | leftpos;
        features[index++] = pw_leftpos;
        long pw_rightw = (pw << 20) | rightw;
        features[index++] = pw_rightw;
        int pw_rightpos = (pw << 10) | rightpos;
        features[index++] = pw_rightpos;
        long pw_leftsiblingw = (pw << 20) | leftsiblingw;
        features[index++] = pw_leftsiblingw;
        int pw_leftsiblingpos = (pw << 10) | leftsiblingpos;
        features[index++] = pw_leftsiblingpos;
        long pw_rightsiblingw = (pw << 20) | rightsiblingw;
        features[index++] = pw_rightsiblingw;
        int pw_rightsiblingpos = (pw << 10) | rightsiblingpos;
        features[index++] = pw_rightsiblingpos;

        //ppos + argument features
        int aw_ppos = (aw << 10) | ppos;
        features[index++] = aw_ppos;
        int ppos_apos = (ppos << 10) | apos;
        features[index++] = ppos_apos;
        int ppos_adeprel = (ppos << 10) | adeprel;
        features[index++] = ppos_adeprel;
        String ppos_deprelpath = ppos + " " + deprelpath;
        features[index++] = ppos_deprelpath;
        String ppos_pospath = ppos + " " + pospath;
        features[index++] = ppos_pospath;
        int ppos_position = (ppos << 2) | position;
        features[index++] = ppos_position;
        int leftw_ppos = (leftw << 10) | ppos;
        features[index++] = leftw_ppos;
        int ppos_leftpos = (ppos << 10) | leftpos;
        features[index++] = ppos_leftpos;
        int rightw_ppos = (rightw << 10) | ppos;
        features[index++] = rightw_ppos;
        int ppos_rightpos = (ppos << 10) | rightpos;
        features[index++] = ppos_rightpos;
        int leftsiblingw_ppos = (leftsiblingw << 10) | ppos;
        features[index++] = leftsiblingw_ppos;
        int ppos_leftsiblingpos = (ppos << 10) | leftsiblingpos;
        features[index++] = ppos_leftsiblingpos;
        int rightsiblingw_ppos = (rightsiblingw << 10) | ppos;
        features[index++] = rightsiblingw_ppos;
        int ppos_rightsiblingpos = (ppos << 10) | rightsiblingpos;
        features[index++] = ppos_rightsiblingpos;

        //pdeprel + argument features
        int aw_pdeprel = (aw << 10) | pdeprel;
        features[index++] = aw_pdeprel;
        int pdeprel_apos = (pdeprel << 10) | apos;
        features[index++] = pdeprel_apos;
        int pdeprel_adeprel = (pdeprel << 10) | adeprel;
        features[index++] = pdeprel_adeprel;
        String pdeprel_deprelpath = pdeprel + " " + deprelpath;
        features[index++] = pdeprel_deprelpath;
        String pdeprel_pospath = pdeprel + " " + pospath;
        features[index++] = pdeprel_pospath;
        int pdeprel_position = (pdeprel << 2) | position;
        features[index++] = pdeprel_position;
        int leftw_pdeprel = (leftw << 10) | pdeprel;
        features[index++] = leftw_pdeprel;
        int pdeprel_leftpos = (pdeprel << 10) | leftpos;
        features[index++] = pdeprel_leftpos;
        int rightw_pdeprel = (rightw << 10) | pdeprel;
        features[index++] = rightw_pdeprel;
        int pdeprel_rightpos = (pdeprel << 10) | rightpos;
        features[index++] = pdeprel_rightpos;
        int leftsiblingw_pdeprel = (leftsiblingw << 10) | pdeprel;
        features[index++] = leftsiblingw_pdeprel;
        int pdeprel_leftsiblingpos = (pdeprel << 10) | leftsiblingpos;
        features[index++] = pdeprel_leftsiblingpos;
        int rightsiblingw_pdeprel = (rightsiblingw << 10) | pdeprel;
        features[index++] = rightsiblingw_pdeprel;
        int pdeprel_rightsiblingpos = (pdeprel << 10) | rightsiblingpos;
        features[index++] = pdeprel_rightsiblingpos;


        //plem + argument features
        long aw_plem = (aw << 20) | plem;
        features[index++] = aw_plem;
        int plem_apos = (plem << 10) | apos;
        features[index++] = plem_apos;
        int plem_adeprel = (plem << 10) | adeprel;
        features[index++] = plem_adeprel;
        String plem_deprelpath = plem + " " + deprelpath;
        features[index++] = plem_deprelpath;
        String plem_pospath = plem + " " + pospath;
        features[index++] = plem_pospath;
        int plem_position = (plem << 2) | position;
        features[index++] = plem_position;
        long leftw_plem = (leftw << 20) | plem;
        features[index++] = leftw_plem;
        int plem_leftpos = (plem << 10) | leftpos;
        features[index++] = plem_leftpos;
        long rightw_plem = (rightw << 20) | plem;
        features[index++] = rightw_plem;
        int plem_rightpos = (plem << 10) | rightpos;
        features[index++] = plem_rightpos;
        long leftsiblingw_plem = (leftsiblingw << 20) | plem;
        features[index++] = leftsiblingw_plem;
        int plem_leftsiblingpos = (plem << 10) | leftsiblingpos;
        features[index++] = plem_leftsiblingpos;
        long rightsiblingw_plem = (rightsiblingw << 20) | plem;
        features[index++] = rightsiblingw_plem;
        int plem_rightsiblingpos = (plem << 10) | rightsiblingpos;
        features[index++] = plem_rightsiblingpos;

        String psense_aw = pSense + " " + aw;
        features[index++] = psense_aw;
        String psense_apos = pSense + " " + apos;
        features[index++] = psense_apos;
        String psense_adeprel = pSense + " " + adeprel;
        features[index++] = psense_adeprel;
        String psense_deprelpath = pSense + " " + deprelpath;
        features[index++] = psense_deprelpath;
        String psense_pospath = pSense + " " + pospath;
        features[index++] = psense_pospath;
        String psense_position = pSense + " " + position;
        features[index++] = psense_position;
        String psense_leftw = pSense + " " + leftw;
        features[index++] = psense_leftw;
        String psense_leftpos = pSense + " " + leftpos;
        features[index++] = psense_leftpos;
        String psense_rightw = pSense + " " + rightw;
        features[index++] = psense_rightw;
        String psense_rightpos = pSense + " " + rightpos;
        features[index++] = psense_rightpos;
        String psense_leftsiblingw = pSense + " " + leftsiblingw;
        features[index++] = psense_leftsiblingw;
        String psense_leftsiblingpos = pSense + " " + leftsiblingpos;
        features[index++] = psense_leftsiblingpos;
        String psense_rightsiblingw = pSense + " " + rightsiblingw;
        features[index++] = psense_rightsiblingw;
        String psense_rightsiblingpos = pSense + " " + rightsiblingpos;
        features[index++] = psense_rightsiblingpos;

        //pprw  + argument features
        long aw_pprw = (aw << 20) | pprw;
        features[index++] = aw_pprw;
        int pprw_apos = (pprw << 10) | apos;
        features[index++] = pprw_apos;
        int pprw_adeprel = (pprw << 10) | adeprel;
        features[index++] = pprw_adeprel;
        String pprw_deprelpath = pprw + " " + deprelpath;
        features[index++] = pprw_deprelpath;
        String pprw_pospath = pprw + " " + pospath;
        features[index++] = pprw_pospath;
        int pprw_position = (pprw << 2) | position;
        features[index++] = pprw_position;
        long leftw_pprw = (leftw << 20) | pprw;
        features[index++] = leftw_pprw;
        int pprw_leftpos = (pprw << 10) | leftpos;
        features[index++] = pprw_leftpos;
        long rightw_pprw = (rightw << 20) | pprw;
        features[index++] = rightw_pprw;
        int pprw_rightpos = (pprw << 10) | rightpos;
        features[index++] = pprw_rightpos;
        long leftsiblingw_pprw = (leftsiblingw << 20) | pprw;
        features[index++] = leftsiblingw_pprw;
        int pprw_leftsiblingpos = (pprw << 10) | leftsiblingpos;
        features[index++] = pprw_leftsiblingpos;
        long rightsiblingw_pprw = (rightsiblingw << 20) | pprw;
        features[index++] = rightsiblingw_pprw;
        int pprw_rightsiblingpos = (pprw << 10) | rightsiblingpos;
        features[index++] = pprw_rightsiblingpos;

        //pdeprel + argument features
        int aw_pprpos = (aw << 10) | pprpos;
        features[index++] = aw_pprpos;
        int pprpos_apos = (pprpos << 10) | apos;
        features[index++] = pprpos_apos;
        int pprpos_adeprel = (pprpos << 10) | adeprel;
        features[index++] = pprpos_adeprel;
        String pprpos_deprelpath = pprpos + " " + deprelpath;
        features[index++] = pprpos_deprelpath;
        String pprpos_pospath = pprpos + " " + pospath;
        features[index++] = pprpos_pospath;
        int pprpos_position = (pprpos << 2) | position;
        features[index++] = pprpos_position;
        int leftw_pprpos = (leftw << 10) | pprpos;
        features[index++] = leftw_pprpos;
        int pprpos_leftpos = (pprpos << 10) | leftpos;
        features[index++] = pprpos_leftpos;
        int rightw_pprpos = (rightw << 10) | pprpos;
        features[index++] = rightw_pprpos;
        int pprpos_rightpos = (pprpos << 10) | rightpos;
        features[index++] = pprpos_rightpos;
        int leftsiblingw_pprpos = (leftsiblingw << 10) | pprpos;
        features[index++] = leftsiblingw_pprpos;
        int pprpos_leftsiblingpos = (pprpos << 10) | leftsiblingpos;
        features[index++] = pprpos_leftsiblingpos;
        int rightsiblingw_pprpos = (rightsiblingw << 10) | pprpos;
        features[index++] = rightsiblingw_pprpos;
        int pprpos_rightsiblingpos = (pprpos << 10) | rightsiblingpos;
        features[index++] = pprpos_rightsiblingpos;

        String pchilddepset_aw = pchilddepset + " " + aw;
        features[index++] = pchilddepset_aw;
        String pchilddepset_apos = pchilddepset + " " + apos;
        features[index++] = pchilddepset_apos;
        String pchilddepset_adeprel = pchilddepset + " " + adeprel;
        features[index++] = pchilddepset_adeprel;
        String pchilddepset_deprelpath = pchilddepset + " " + deprelpath;
        features[index++] = pchilddepset_deprelpath;
        String pchilddepset_pospath = pchilddepset + " " + pospath;
        features[index++] = pchilddepset_pospath;
        String pchilddepset_position = pchilddepset + " " + position;
        features[index++] = pchilddepset_position;
        String pchilddepset_leftw = pchilddepset + " " + leftw;
        features[index++] = pchilddepset_leftw;
        String pchilddepset_leftpos = pchilddepset + " " + leftpos;
        features[index++] = pchilddepset_leftpos;
        String pchilddepset_rightw = pchilddepset + " " + rightw;
        features[index++] = pchilddepset_rightw;
        String pchilddepset_rightpos = pchilddepset + " " + rightpos;
        features[index++] = pchilddepset_rightpos;
        String pchilddepset_leftsiblingw = pchilddepset + " " + leftsiblingw;
        features[index++] = pchilddepset_leftsiblingw;
        String pchilddepset_leftsiblingpos = pchilddepset + " " + leftsiblingpos;
        features[index++] = pchilddepset_leftsiblingpos;
        String pchilddepset_rightsiblingw = pchilddepset + " " + rightsiblingw;
        features[index++] = pchilddepset_rightsiblingw;
        String pchilddepset_rightsiblingpos = pchilddepset + " " + rightsiblingpos;
        features[index++] = pchilddepset_rightsiblingpos;

        String pdepsubcat_aw = pdepsubcat + " " + aw;
        features[index++] = pdepsubcat_aw;
        String pdepsubcat_apos = pdepsubcat + " " + apos;
        features[index++] = pdepsubcat_apos;
        String pdepsubcat_adeprel = pdepsubcat + " " + adeprel;
        features[index++] = pdepsubcat_adeprel;
        String pdepsubcat_deprelpath = pdepsubcat + " " + deprelpath;
        features[index++] = pdepsubcat_deprelpath;
        String pdepsubcat_pospath = pdepsubcat + " " + pospath;
        features[index++] = pdepsubcat_pospath;
        String pdepsubcat_position = pdepsubcat + " " + position;
        features[index++] = pdepsubcat_position;
        String pdepsubcat_leftw = pdepsubcat + " " + leftw;
        features[index++] = pdepsubcat_leftw;
        String pdepsubcat_leftpos = pdepsubcat + " " + leftpos;
        features[index++] = pdepsubcat_leftpos;
        String pdepsubcat_rightw = pdepsubcat + " " + rightw;
        features[index++] = pdepsubcat_rightw;
        String pdepsubcat_rightpos = pdepsubcat + " " + rightpos;
        features[index++] = pdepsubcat_rightpos;
        String pdepsubcat_leftsiblingw = pdepsubcat + " " + leftsiblingw;
        features[index++] = pdepsubcat_leftsiblingw;
        String pdepsubcat_leftsiblingpos = pdepsubcat + " " + leftsiblingpos;
        features[index++] = pdepsubcat_leftsiblingpos;
        String pdepsubcat_rightsiblingw = pdepsubcat + " " + rightsiblingw;
        features[index++] = pdepsubcat_rightsiblingw;
        String pdepsubcat_rightsiblingpos = pdepsubcat + " " + rightsiblingpos;
        features[index++] = pdepsubcat_rightsiblingpos;

        String pchildposset_aw = pchildposset + " " + aw;
        features[index++] = pchildposset_aw;
        String pchildposset_apos = pchildposset + " " + apos;
        features[index++] = pchildposset_apos;
        String pchildposset_adeprel = pchildposset + " " + adeprel;
        features[index++] = pchildposset_adeprel;
        String pchildposset_deprelpath = pchildposset + " " + deprelpath;
        features[index++] = pchildposset_deprelpath;
        String pchildposset_pospath = pchildposset + " " + pospath;
        features[index++] = pchildposset_pospath;
        String pchildposset_position = pchildposset + " " + position;
        features[index++] = pchildposset_position;
        String pchildposset_leftw = pchildposset + " " + leftw;
        features[index++] = pchildposset_leftw;
        String pchildposset_leftpos = pchildposset + " " + leftpos;
        features[index++] = pchildposset_leftpos;
        String pchildposset_rightw = pchildposset + " " + rightw;
        features[index++] = pchildposset_rightw;
        String pchildposset_rightpos = pchildposset + " " + rightpos;
        features[index++] = pchildposset_rightpos;
        String pchildposset_leftsiblingw = pchildposset + " " + leftsiblingw;
        features[index++] = pchildposset_leftsiblingw;
        String pchildposset_leftsiblingpos = pchildposset + " " + leftsiblingpos;
        features[index++] = pchildposset_leftsiblingpos;
        String pchildposset_rightsiblingw = pchildposset + " " + rightsiblingw;
        features[index++] = pchildposset_rightsiblingw;
        String pchildposset_rightsiblingpos = pchildposset + " " + rightsiblingpos;
        features[index++] = pchildposset_rightsiblingpos;


        //pchildwset + argument features
        String pchildwset_aw = pchildwset + " " + aw;
        features[index++] = pchildwset_aw;
        String pchildwset_apos = pchildwset + " " + apos;
        features[index++] = pchildwset_apos;
        String pchildwset_adeprel = pchildwset + " " + adeprel;
        features[index++] = pchildwset_adeprel;
        String pchildwset_deprelpath = pchildwset + " " + deprelpath;
        features[index++] = pchildwset_deprelpath;
        String pchildwset_pospath = pchildwset + " " + pospath;
        features[index++] = pchildwset_pospath;
        String pchildwset_position = pchildwset + " " + position;
        features[index++] = pchildwset_position;
        String pchildwset_leftw = pchildwset + " " + leftw;
        features[index++] = pchildwset_leftw;
        String pchildwset_leftpos = pchildwset + " " + leftpos;
        features[index++] = pchildwset_leftpos;
        String pchildwset_rightw = pchildwset + " " + rightw;
        features[index++] = pchildwset_rightw;
        String pchildwset_rightpos = pchildwset + " " + rightpos;
        features[index++] = pchildwset_rightpos;
        String pchildwset_leftsiblingw = pchildwset + " " + leftsiblingw;
        features[index++] = pchildwset_leftsiblingw;
        String pchildwset_leftsiblingpos = pchildwset + " " + leftsiblingpos;
        features[index++] = pchildwset_leftsiblingpos;
        String pchildwset_rightsiblingw = pchildwset + " " + rightsiblingw;
        features[index++] = pchildwset_rightsiblingw;
        String pchildwset_rightsiblingpos = pchildwset + " " + rightsiblingpos;
        features[index++] = pchildwset_rightsiblingpos;
        return features;
    }

    public static Object[] extractJointFeatures(int pIdx, String pSense, int aIdx, Sentence sentence, int length,
                                                IndexMap indexMap) throws Exception {

        // todo object; int; respectively
        Object[] features = new Object[length];
        int[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        int[] sentenceWords = sentence.getWords();
        int[] sentencePOSTags = sentence.getPosTags();
        int[] sentenceLemmas = sentence.getLemmas();
        TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();

        //predicate features
        int pw = sentenceWords[pIdx];
        int ppos = sentencePOSTags[pIdx];
        int plem = sentenceLemmas[pIdx];
        int pdeprel = sentenceDepLabels[pIdx];
        int pprw = sentenceWords[sentenceDepHeads[pIdx]];
        int pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
        String pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags, indexMap);
        String pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags, indexMap);


        int leftMostDependentIndex = getLeftMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int rightMostDependentIndex = getRightMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int lefSiblingIndex = getLeftSiblingIndex(aIdx, pIdx, sentenceReverseDepHeads);
        int rightSiblingIndex = getRightSiblingIndex(aIdx, pIdx, sentenceReverseDepHeads);

        //argument features
        int aw = sentenceWords[aIdx];
        int apos = sentencePOSTags[aIdx];
        int adeprel = sentenceDepLabels[aIdx];

        //predicate-argument features
        String deprelpath = StringUtils.convertPathArrayIntoString(sentence.getDepPath(pIdx, aIdx));
        String pospath = StringUtils.convertPathArrayIntoString(sentence.getPOSPath(pIdx, aIdx));

        int position = 0; //on
        if (pIdx < aIdx)
            position = 1; //before
        else
            position = 2; //after

        int leftw = (leftMostDependentIndex != -1) ? sentenceWords[leftMostDependentIndex] : indexMap.getNullIdx();
        int leftpos = (leftMostDependentIndex != -1) ? sentencePOSTags[leftMostDependentIndex] : indexMap.getNullIdx();
        int rightw = (rightMostDependentIndex != -1) ? sentenceWords[rightMostDependentIndex] : indexMap.getNullIdx();
        int rightpos = (rightMostDependentIndex != -1) ? sentencePOSTags[rightMostDependentIndex] : indexMap.getNullIdx();
        int rightsiblingw = (rightSiblingIndex != -1) ? sentenceWords[rightSiblingIndex] : indexMap.getNullIdx();
        int rightsiblingpos = (rightSiblingIndex != -1) ? sentencePOSTags[rightSiblingIndex] : indexMap.getNullIdx();
        int leftsiblingw = (lefSiblingIndex != -1) ? sentenceWords[lefSiblingIndex] : indexMap.getNullIdx();
        int leftsiblingpos = (lefSiblingIndex != -1) ? sentencePOSTags[lefSiblingIndex] : indexMap.getNullIdx();

        //build feature vector for argument identification/classification modules
        int index = 0;

        //////////////////////////////////////////////////////////////////////
        ////////////////////////// SINGLE FEATURES //////////////////////////
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////
        ////// PREDICATE FEATURES //////
        ////////////////////////////////
        features[index++] = pw;
        features[index++] = ppos;
        features[index++] = plem;
        features[index++] = pdeprel;
        features[index++] = pSense;
        features[index++] = pprw;
        features[index++] = pprpos;
        features[index++] = pdepsubcat;
        features[index++] = pchilddepset;
        features[index++] = pchildposset;
        features[index++] = pchildwset;

        ////////////////////////////////
        ////// ARGUMENT FEATURES //////
        ////////////////////////////////
        features[index++] = aw;
        features[index++] = apos;
        features[index++] = adeprel;
        features[index++] = deprelpath;
        features[index++] = pospath;
        features[index++] = position;
        features[index++] = leftw;
        features[index++] = leftpos;
        features[index++] = rightw;
        features[index++] = rightpos;
        features[index++] = leftsiblingw;
        features[index++] = leftsiblingpos;
        features[index++] = rightsiblingw;
        features[index++] = rightsiblingpos;

        //////////////////////////////////////////////////////////////////////
        ///////////////// PREDICATE-ARGUMENT CONJOINED FEATURES //////////////
        //////////////////////////////////////////////////////////////////////

        // todo 20 bits for words, 10 bits for pos, 10 bits for dependencies
        // todo e.g. (pw<<20) | aw ==> 20+20> 32 ==> long
        // pw + argument features
        long pw_aw = (pw << 20) | aw;
        features[index++] = pw_aw;
        // todo (pw<<10) | pos ==> 10+20<32 ==> int
        int pw_apos = (pw << 10) | apos;
        features[index++] = pw_apos;
        int pw_adeprel = (pw << 10) | adeprel;
        features[index++] = pw_adeprel;
        String pw_deprelpath = pw + " " + deprelpath;
        features[index++] = pw_deprelpath;
        String pw_pospath = pw + " " + pospath;
        features[index++] = pw_pospath;
        int pw_position = (pw << 2) | position;
        features[index++] = pw_position;
        long pw_leftw = (pw << 20) | leftw;
        features[index++] = pw_leftw;
        int pw_leftpos = (pw << 10) | leftpos;
        features[index++] = pw_leftpos;
        long pw_rightw = (pw << 20) | rightw;
        features[index++] = pw_rightw;
        int pw_rightpos = (pw << 10) | rightpos;
        features[index++] = pw_rightpos;
        long pw_leftsiblingw = (pw << 20) | leftsiblingw;
        features[index++] = pw_leftsiblingw;
        int pw_leftsiblingpos = (pw << 10) | leftsiblingpos;
        features[index++] = pw_leftsiblingpos;
        long pw_rightsiblingw = (pw << 20) | rightsiblingw;
        features[index++] = pw_rightsiblingw;
        int pw_rightsiblingpos = (pw << 10) | rightsiblingpos;
        features[index++] = pw_rightsiblingpos;

        //ppos + argument features
        int aw_ppos = (aw << 10) | ppos;
        features[index++] = aw_ppos;
        int ppos_apos = (ppos << 10) | apos;
        features[index++] = ppos_apos;
        int ppos_adeprel = (ppos << 10) | adeprel;
        features[index++] = ppos_adeprel;
        String ppos_deprelpath = ppos + " " + deprelpath;
        features[index++] = ppos_deprelpath;
        String ppos_pospath = ppos + " " + pospath;
        features[index++] = ppos_pospath;
        int ppos_position = (ppos << 2) | position;
        features[index++] = ppos_position;
        int leftw_ppos = (leftw << 10) | ppos;
        features[index++] = leftw_ppos;
        int ppos_leftpos = (ppos << 10) | leftpos;
        features[index++] = ppos_leftpos;
        int rightw_ppos = (rightw << 10) | ppos;
        features[index++] = rightw_ppos;
        int ppos_rightpos = (ppos << 10) | rightpos;
        features[index++] = ppos_rightpos;
        int leftsiblingw_ppos = (leftsiblingw << 10) | ppos;
        features[index++] = leftsiblingw_ppos;
        int ppos_leftsiblingpos = (ppos << 10) | leftsiblingpos;
        features[index++] = ppos_leftsiblingpos;
        int rightsiblingw_ppos = (rightsiblingw << 10) | ppos;
        features[index++] = rightsiblingw_ppos;
        int ppos_rightsiblingpos = (ppos << 10) | rightsiblingpos;
        features[index++] = ppos_rightsiblingpos;


        //pdeprel + argument features
        int aw_pdeprel = (aw << 10) | pdeprel;
        features[index++] = aw_pdeprel;
        int pdeprel_apos = (pdeprel << 10) | apos;
        features[index++] = pdeprel_apos;
        int pdeprel_adeprel = (pdeprel << 10) | adeprel;
        features[index++] = pdeprel_adeprel;
        String pdeprel_deprelpath = pdeprel + " " + deprelpath;
        features[index++] = pdeprel_deprelpath;
        String pdeprel_pospath = pdeprel + " " + pospath;
        features[index++] = pdeprel_pospath;
        int pdeprel_position = (pdeprel << 2) | position;
        features[index++] = pdeprel_position;
        int leftw_pdeprel = (leftw << 10) | pdeprel;
        features[index++] = leftw_pdeprel;
        int pdeprel_leftpos = (pdeprel << 10) | leftpos;
        features[index++] = pdeprel_leftpos;
        int rightw_pdeprel = (rightw << 10) | pdeprel;
        features[index++] = rightw_pdeprel;
        int pdeprel_rightpos = (pdeprel << 10) | rightpos;
        features[index++] = pdeprel_rightpos;
        int leftsiblingw_pdeprel = (leftsiblingw << 10) | pdeprel;
        features[index++] = leftsiblingw_pdeprel;
        int pdeprel_leftsiblingpos = (pdeprel << 10) | leftsiblingpos;
        features[index++] = pdeprel_leftsiblingpos;
        int rightsiblingw_pdeprel = (rightsiblingw << 10) | pdeprel;
        features[index++] = rightsiblingw_pdeprel;
        int pdeprel_rightsiblingpos = (pdeprel << 10) | rightsiblingpos;
        features[index++] = pdeprel_rightsiblingpos;


        //plem + argument features
        long aw_plem = (aw << 20) | plem;
        features[index++] = aw_plem;
        int plem_apos = (plem << 10) | apos;
        features[index++] = plem_apos;
        int plem_adeprel = (plem << 10) | adeprel;
        features[index++] = plem_adeprel;
        String plem_deprelpath = plem + " " + deprelpath;
        features[index++] = plem_deprelpath;
        String plem_pospath = plem + " " + pospath;
        features[index++] = plem_pospath;
        int plem_position = (plem << 2) | position;
        features[index++] = plem_position;
        long leftw_plem = (leftw << 20) | plem;
        features[index++] = leftw_plem;
        int plem_leftpos = (plem << 10) | leftpos;
        features[index++] = plem_leftpos;
        long rightw_plem = (rightw << 20) | plem;
        features[index++] = rightw_plem;
        int plem_rightpos = (plem << 10) | rightpos;
        features[index++] = plem_rightpos;
        long leftsiblingw_plem = (leftsiblingw << 20) | plem;
        features[index++] = leftsiblingw_plem;
        int plem_leftsiblingpos = (plem << 10) | leftsiblingpos;
        features[index++] = plem_leftsiblingpos;
        long rightsiblingw_plem = (rightsiblingw << 20) | plem;
        features[index++] = rightsiblingw_plem;
        int plem_rightsiblingpos = (plem << 10) | rightsiblingpos;
        features[index++] = plem_rightsiblingpos;

        String psense_aw = pSense + " " + aw;
        features[index++] = psense_aw;
        String psense_apos = pSense + " " + apos;
        features[index++] = psense_apos;
        String psense_adeprel = pSense + " " + adeprel;
        features[index++] = psense_adeprel;
        String psense_deprelpath = pSense + " " + deprelpath;
        features[index++] = psense_deprelpath;
        String psense_pospath = pSense + " " + pospath;
        features[index++] = psense_pospath;
        String psense_position = pSense + " " + position;
        features[index++] = psense_position;
        String psense_leftw = pSense + " " + leftw;
        features[index++] = psense_leftw;
        String psense_leftpos = pSense + " " + leftpos;
        features[index++] = psense_leftpos;
        String psense_rightw = pSense + " " + rightw;
        features[index++] = psense_rightw;
        String psense_rightpos = pSense + " " + rightpos;
        features[index++] = psense_rightpos;
        String psense_leftsiblingw = pSense + " " + leftsiblingw;
        features[index++] = psense_leftsiblingw;
        String psense_leftsiblingpos = pSense + " " + leftsiblingpos;
        features[index++] = psense_leftsiblingpos;
        String psense_rightsiblingw = pSense + " " + rightsiblingw;
        features[index++] = psense_rightsiblingw;
        String psense_rightsiblingpos = pSense + " " + rightsiblingpos;
        features[index++] = psense_rightsiblingpos;


        //pprw  + argument features
        long aw_pprw = (aw << 20) | pprw;
        features[index++] = aw_pprw;
        int pprw_apos = (pprw << 10) | apos;
        features[index++] = pprw_apos;
        int pprw_adeprel = (pprw << 10) | adeprel;
        features[index++] = pprw_adeprel;
        String pprw_deprelpath = pprw + " " + deprelpath;
        features[index++] = pprw_deprelpath;
        String pprw_pospath = pprw + " " + pospath;
        features[index++] = pprw_pospath;
        int pprw_position = (pprw << 2) | position;
        features[index++] = pprw_position;
        long leftw_pprw = (leftw << 20) | pprw;
        features[index++] = leftw_pprw;
        int pprw_leftpos = (pprw << 10) | leftpos;
        features[index++] = pprw_leftpos;
        long rightw_pprw = (rightw << 20) | pprw;
        features[index++] = rightw_pprw;
        int pprw_rightpos = (pprw << 10) | rightpos;
        features[index++] = pprw_rightpos;
        long leftsiblingw_pprw = (leftsiblingw << 20) | pprw;
        features[index++] = leftsiblingw_pprw;
        int pprw_leftsiblingpos = (pprw << 10) | leftsiblingpos;
        features[index++] = pprw_leftsiblingpos;
        long rightsiblingw_pprw = (rightsiblingw << 20) | pprw;
        features[index++] = rightsiblingw_pprw;
        int pprw_rightsiblingpos = (pprw << 10) | rightsiblingpos;
        features[index++] = pprw_rightsiblingpos;

        //pdeprel + argument features
        int aw_pprpos = (aw << 10) | pprpos;
        features[index++] = aw_pprpos;
        int pprpos_apos = (pprpos << 10) | apos;
        features[index++] = pprpos_apos;
        int pprpos_adeprel = (pprpos << 10) | adeprel;
        features[index++] = pprpos_adeprel;
        String pprpos_deprelpath = pprpos + " " + deprelpath;
        features[index++] = pprpos_deprelpath;
        String pprpos_pospath = pprpos + " " + pospath;
        features[index++] = pprpos_pospath;
        int pprpos_position = (pprpos << 2) | position;
        features[index++] = pprpos_position;
        int leftw_pprpos = (leftw << 10) | pprpos;
        features[index++] = leftw_pprpos;
        int pprpos_leftpos = (pprpos << 10) | leftpos;
        features[index++] = pprpos_leftpos;
        int rightw_pprpos = (rightw << 10) | pprpos;
        features[index++] = rightw_pprpos;
        int pprpos_rightpos = (pprpos << 10) | rightpos;
        features[index++] = pprpos_rightpos;
        int leftsiblingw_pprpos = (leftsiblingw << 10) | pprpos;
        features[index++] = leftsiblingw_pprpos;
        int pprpos_leftsiblingpos = (pprpos << 10) | leftsiblingpos;
        features[index++] = pprpos_leftsiblingpos;
        int rightsiblingw_pprpos = (rightsiblingw << 10) | pprpos;
        features[index++] = rightsiblingw_pprpos;
        int pprpos_rightsiblingpos = (pprpos << 10) | rightsiblingpos;
        features[index++] = pprpos_rightsiblingpos;

        //pchilddepset + argument features
        String pchilddepset_aw = pchilddepset + " " + aw;
        features[index++] = pchilddepset_aw;
        String pchilddepset_apos = pchilddepset + " " + apos;
        features[index++] = pchilddepset_apos;
        String pchilddepset_adeprel = pchilddepset + " " + adeprel;
        features[index++] = pchilddepset_adeprel;
        String pchilddepset_deprelpath = pchilddepset + " " + deprelpath;
        features[index++] = pchilddepset_deprelpath;
        String pchilddepset_pospath = pchilddepset + " " + pospath;
        features[index++] = pchilddepset_pospath;
        String pchilddepset_position = pchilddepset + " " + position;
        features[index++] = pchilddepset_position;
        String pchilddepset_leftw = pchilddepset + " " + leftw;
        features[index++] = pchilddepset_leftw;
        String pchilddepset_leftpos = pchilddepset + " " + leftpos;
        features[index++] = pchilddepset_leftpos;
        String pchilddepset_rightw = pchilddepset + " " + rightw;
        features[index++] = pchilddepset_rightw;
        String pchilddepset_rightpos = pchilddepset + " " + rightpos;
        features[index++] = pchilddepset_rightpos;
        String pchilddepset_leftsiblingw = pchilddepset + " " + leftsiblingw;
        features[index++] = pchilddepset_leftsiblingw;
        String pchilddepset_leftsiblingpos = pchilddepset + " " + leftsiblingpos;
        features[index++] = pchilddepset_leftsiblingpos;
        String pchilddepset_rightsiblingw = pchilddepset + " " + rightsiblingw;
        features[index++] = pchilddepset_rightsiblingw;
        String pchilddepset_rightsiblingpos = pchilddepset + " " + rightsiblingpos;
        features[index++] = pchilddepset_rightsiblingpos;

        String pdepsubcat_aw = pdepsubcat + " " + aw;
        features[index++] = pdepsubcat_aw;
        String pdepsubcat_apos = pdepsubcat + " " + apos;
        features[index++] = pdepsubcat_apos;
        String pdepsubcat_adeprel = pdepsubcat + " " + adeprel;
        features[index++] = pdepsubcat_adeprel;
        String pdepsubcat_deprelpath = pdepsubcat + " " + deprelpath;
        features[index++] = pdepsubcat_deprelpath;
        String pdepsubcat_pospath = pdepsubcat + " " + pospath;
        features[index++] = pdepsubcat_pospath;
        String pdepsubcat_position = pdepsubcat + " " + position;
        features[index++] = pdepsubcat_position;
        String pdepsubcat_leftw = pdepsubcat + " " + leftw;
        features[index++] = pdepsubcat_leftw;
        String pdepsubcat_leftpos = pdepsubcat + " " + leftpos;
        features[index++] = pdepsubcat_leftpos;
        String pdepsubcat_rightw = pdepsubcat + " " + rightw;
        features[index++] = pdepsubcat_rightw;
        String pdepsubcat_rightpos = pdepsubcat + " " + rightpos;
        features[index++] = pdepsubcat_rightpos;
        String pdepsubcat_leftsiblingw = pdepsubcat + " " + leftsiblingw;
        features[index++] = pdepsubcat_leftsiblingw;
        String pdepsubcat_leftsiblingpos = pdepsubcat + " " + leftsiblingpos;
        features[index++] = pdepsubcat_leftsiblingpos;
        String pdepsubcat_rightsiblingw = pdepsubcat + " " + rightsiblingw;
        features[index++] = pdepsubcat_rightsiblingw;
        String pdepsubcat_rightsiblingpos = pdepsubcat + " " + rightsiblingpos;
        features[index++] = pdepsubcat_rightsiblingpos;

        //pchildposset + argument features
        String pchildposset_aw = pchildposset + " " + aw;
        features[index++] = pchildposset_aw;
        String pchildposset_apos = pchildposset + " " + apos;
        features[index++] = pchildposset_apos;
        String pchildposset_adeprel = pchildposset + " " + adeprel;
        features[index++] = pchildposset_adeprel;
        String pchildposset_deprelpath = pchildposset + " " + deprelpath;
        features[index++] = pchildposset_deprelpath;
        String pchildposset_pospath = pchildposset + " " + pospath;
        features[index++] = pchildposset_pospath;
        String pchildposset_position = pchildposset + " " + position;
        features[index++] = pchildposset_position;
        String pchildposset_leftw = pchildposset + " " + leftw;
        features[index++] = pchildposset_leftw;
        String pchildposset_leftpos = pchildposset + " " + leftpos;
        features[index++] = pchildposset_leftpos;
        String pchildposset_rightw = pchildposset + " " + rightw;
        features[index++] = pchildposset_rightw;
        String pchildposset_rightpos = pchildposset + " " + rightpos;
        features[index++] = pchildposset_rightpos;
        String pchildposset_leftsiblingw = pchildposset + " " + leftsiblingw;
        features[index++] = pchildposset_leftsiblingw;
        String pchildposset_leftsiblingpos = pchildposset + " " + leftsiblingpos;
        features[index++] = pchildposset_leftsiblingpos;
        String pchildposset_rightsiblingw = pchildposset + " " + rightsiblingw;
        features[index++] = pchildposset_rightsiblingw;
        String pchildposset_rightsiblingpos = pchildposset + " " + rightsiblingpos;
        features[index++] = pchildposset_rightsiblingpos;

        //pchildwset + argument features
        String pchildwset_aw = pchildwset + " " + aw;
        features[index++] = pchildwset_aw;
        String pchildwset_apos = pchildwset + " " + apos;
        features[index++] = pchildwset_apos;
        String pchildwset_adeprel = pchildwset + " " + adeprel;
        features[index++] = pchildwset_adeprel;
        String pchildwset_deprelpath = pchildwset + " " + deprelpath;
        features[index++] = pchildwset_deprelpath;
        String pchildwset_pospath = pchildwset + " " + pospath;
        features[index++] = pchildwset_pospath;
        String pchildwset_position = pchildwset + " " + position;
        features[index++] = pchildwset_position;
        String pchildwset_leftw = pchildwset + " " + leftw;
        features[index++] = pchildwset_leftw;
        String pchildwset_leftpos = pchildwset + " " + leftpos;
        features[index++] = pchildwset_leftpos;
        String pchildwset_rightw = pchildwset + " " + rightw;
        features[index++] = pchildwset_rightw;
        String pchildwset_rightpos = pchildwset + " " + rightpos;
        features[index++] = pchildwset_rightpos;
        String pchildwset_leftsiblingw = pchildwset + " " + leftsiblingw;
        features[index++] = pchildwset_leftsiblingw;
        String pchildwset_leftsiblingpos = pchildwset + " " + leftsiblingpos;
        features[index++] = pchildwset_leftsiblingpos;
        String pchildwset_rightsiblingw = pchildwset + " " + rightsiblingw;
        features[index++] = pchildwset_rightsiblingw;
        String pchildwset_rightsiblingpos = pchildwset + " " + rightsiblingpos;
        features[index++] = pchildwset_rightsiblingpos;
        return features;
    }

    private static String getDepSubCat(int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads,
                                       int[] sentenceDepLabels, int[] posTags, IndexMap indexMap) throws Exception {
        StringBuilder subCat = new StringBuilder();
        if (sentenceReverseDepHeads[pIdx] != null) {
            //predicate has >1 children
            for (int child : sentenceReverseDepHeads[pIdx]) {
                String pos = indexMap.int2str(posTags[child]);
                if (!punctuations.contains(pos)) {
                    subCat.append(sentenceDepLabels[child]);
                    subCat.append("\t");
                }
            }
        }
        return subCat.toString().trim();
    }

    private static String getChildSet(int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads,
                                      int[] collection, int[] posTags, IndexMap map) throws Exception {
        StringBuilder childSet = new StringBuilder();
        TreeSet<Integer> children = new TreeSet<Integer>();
        if (sentenceReverseDepHeads[pIdx] != null) {
            for (int child : sentenceReverseDepHeads[pIdx]) {
                String pos = map.int2str(posTags[child]);
                if (!punctuations.contains(pos)) {
                    children.add(collection[child]);
                }
            }
        }
        for (int child : children) {
            childSet.append(child);
            childSet.append("\t");
        }
        return childSet.toString().trim();
    }

    private static int getLeftMostDependentIndex(int aIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads[aIdx] != null) {
            //this argument has at least one child
            return sentenceReverseDepHeads[aIdx].first();
        }
        return -1;
    }

    private static int getRightMostDependentIndex(int aIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads[aIdx] != null) {
            return sentenceReverseDepHeads[aIdx].last();
        }
        return -1;
    }

    private static int getLeftSiblingIndex(int aIdx, int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        TreeSet<Integer> argSiblings = new TreeSet<Integer>();
        if (sentenceReverseDepHeads[pIdx] != null) {
            argSiblings = sentenceReverseDepHeads[pIdx];
        }

        if (argSiblings.lower(aIdx) != null)
            return argSiblings.lower(aIdx);
        return -1;
    }

    private static int getRightSiblingIndex(int aIdx, int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        TreeSet<Integer> argSiblings = new TreeSet<Integer>();
        if (sentenceReverseDepHeads[pIdx] != null)
            argSiblings = sentenceReverseDepHeads[pIdx];

        if (argSiblings.higher(aIdx) != null)
            return argSiblings.higher(aIdx);
        return -1;
    }

}
