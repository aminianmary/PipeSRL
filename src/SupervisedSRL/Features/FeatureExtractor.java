package SupervisedSRL.Features;
/**
 * Created by Maryam Aminian on 5/17/16.
 */

import Sentence.Sentence;
import Sentence.Predicate;
import Sentence.Argument;
import SupervisedSRL.Strcutures.IndexMap;
import apple.laf.JRSUIUtils;
import util.StringUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.Set;

public class FeatureExtractor {

    // todo Object[]
    public static Object[]  extractFeatures(int pIdx, String pSense, int aIdx, Sentence sentence, String state, int length,
                                           IndexMap indexMap) {

        // todo object; int; respectively
        Object[] features = new Object [length];
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
        String pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels);
        String pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels);
        String pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags);
        String pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords);

       // String voice = sentence.getVoice(pIdx);

        if (state.equals("AI")) {
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
            int aw_position = (aw<<2) | position;
            features[index++] = aw_position;
            String psense_position = pSense + " " +position;
            features[index++] = psense_position;
            String psense_apos = pSense +" "+apos;
            features[index++] = psense_apos;
            String deprelpath_position = deprelpath + " " +position;
            features[index++] = deprelpath_position;
            int pprw_adeprel = (pprw<<10) | adeprel;
            features[index++] = pprw_adeprel;
            int pw_leftpos = (pw<<10) | leftpos;
            features[index++] = pw_leftpos;
            String psense_pchildwset = pSense + " " + pchildwset;
            features[index++] = psense_pchildwset;
            String adeprel_deprelpath = adeprel+" "+ deprelpath;
            features[index++] = adeprel_deprelpath;
            String pospath_rightsiblingpos = pospath+" "+rightsiblingpos;
            features[index++] = pospath_rightsiblingpos;
            String pchilddepset_adeprel = pchilddepset +" "+ adeprel;
            features[index++] = pchilddepset_adeprel;
            int apos_adeprel = (apos<<10) | adeprel;
            features[index++] = apos_adeprel;
            int leftpos_rightpos = (leftpos<<10) | rightpos;
            features[index++] = leftpos_rightpos;
            int pdeprel_adeprel = (pdeprel<<10) | adeprel;
            features[index++] = pdeprel_adeprel;


            //////////////////////////////////////////////////////////////////////
            ///////////////// PREDICATE-ARGUMENT CONJOINED FEATURES (154) ////////
            //////////////////////////////////////////////////////////////////////
            /*
            // todo 20 bits for words, 10 bits for pos, 10 bits for dependencies
            // todo e.g. (pw<<20) | aw ==> 20+20> 32 ==> long
            // pw + argument features
            long pw_aw = (pw<<20) | aw;
            features[index++] = pw_aw;
            // todo (pw<<10) | pos ==> 10+20<32 ==> int
            int pw_apos = (pw<<10) | apos ;
            features[index++] = pw_apos;
            int pw_adeprel = (pw<<10) | adeprel;
            features[index++] = pw_adeprel;
            //StringBuilder pw_deprelpath= new StringBuilder();
            //pw_deprelpath.append(pw);
            //pw_deprelpath.append(" ");
            //pw_deprelpath.append(deprelpath);
            String pw_deprelpath = pw +" "+deprelpath;
            features[index++] = pw_deprelpath;
            //StringBuilder pw_pospath= new StringBuilder();
            //pw_deprelpath.append(pw);
            //pw_deprelpath.append(" ");
            //pw_deprelpath.append(pospath);
            String pw_pospath = pw +" "+pospath;
            features[index++] = pw_pospath;
            int pw_position = (pw<<2) | position;
            features[index++] = pw_position;
            long pw_leftw = (pw<<20) | leftw;
            features[index++] = pw_leftw;
            int pw_leftpos = (pw<<10) | leftpos;
            features[index++] = pw_leftpos;
            long pw_rightw = (pw<<20) | rightw;
            features[index++] = pw_rightw;
            int pw_rightpos = (pw<<10) | rightpos;
            features[index++] = pw_rightpos;
            long pw_leftsiblingw = (pw<<20) | leftsiblingw;
            features[index++] = pw_leftsiblingw;
            int pw_leftsiblingpos= (pw<<10) | leftsiblingpos;
            features[index++] = pw_leftsiblingpos;
            long pw_rightsiblingw = (pw<<20) | rightsiblingw;
            features[index++] = pw_rightsiblingw;
            int pw_rightsiblingpos = (pw<<10) | rightsiblingpos;
            features[index++] = pw_rightsiblingpos;

            //ppos + argument features
            int aw_ppos = (aw<<10) | ppos;
            features[index++] = aw_ppos;
            int ppos_apos = (ppos<<10) | apos;
            features[index++] = ppos_apos;
            int ppos_adeprel = (ppos<<10) | adeprel;
            features[index++] = ppos_adeprel;
            //StringBuilder ppos_deprelpath = new StringBuilder();
            //ppos_deprelpath.append(ppos);
            //ppos_deprelpath.append(" ");
            //ppos_deprelpath.append(deprelpath);
            String ppos_deprelpath = ppos+" "+deprelpath;
            features[index++] = ppos_deprelpath;
            //StringBuilder ppos_pospath = new StringBuilder();
            //ppos_pospath.append(ppos);
            //ppos_pospath.append(" ");
            //ppos_pospath.append(pospath);
            String ppos_pospath= ppos +" "+pospath;
            features[index++] = ppos_pospath;
            int ppos_position = (ppos<<2) | position;
            features[index++] = ppos_position;
            int leftw_ppos = (leftw<<10) | ppos;
            features[index++] = leftw_ppos;
            int ppos_leftpos = (ppos<<10) | leftpos;
            features[index++] = ppos_leftpos;
            int rightw_ppos = (rightw<<10) | ppos;
            features[index++] = rightw_ppos;
            int ppos_rightpos = (ppos<<10) | rightpos;
            features[index++] = ppos_rightpos;
            int leftsiblingw_ppos = (leftsiblingw<<10) | ppos;
            features[index++] = leftsiblingw_ppos;
            int ppos_leftsiblingpos = (ppos<<10) | leftsiblingpos;
            features[index++] = ppos_leftsiblingpos;
            int rightsiblingw_ppos = (rightsiblingw<<10) | ppos;
            features[index++] = rightsiblingw_ppos;
            int ppos_rightsiblingpos = (ppos<<10) | rightsiblingpos;
            features[index++] = ppos_rightsiblingpos;


            //pdeprel + argument features
            int aw_pdeprel = (aw<<10) | pdeprel;
            features[index++] = aw_pdeprel;
            int pdeprel_apos = (pdeprel<<10) | apos;
            features[index++] = pdeprel_apos;
            int pdeprel_adeprel = (pdeprel<<10) | adeprel;
            features[index++] = pdeprel_adeprel;
            //StringBuilder pdeprel_deprelpath = new StringBuilder();
            //pdeprel_deprelpath.append(pdeprel);
            //pdeprel_deprelpath.append(" ");
            //pdeprel_deprelpath.append(deprelpath);
            String pdeprel_deprelpath = pdeprel + " "+deprelpath;
            features[index++] = pdeprel_deprelpath;
            //StringBuilder pdeprel_pospath= new StringBuilder();
            //pdeprel_pospath.append(pdeprel);
            //pdeprel_pospath.append(" ");
            //pdeprel_pospath.append(pospath);
            String pdeprel_pospath = pdeprel +" "+pospath;
            features[index++] = pdeprel_pospath;
            int pdeprel_position = (pdeprel<<2) | position;
            features[index++] = pdeprel_position;
            int leftw_pdeprel = (leftw<<10) | pdeprel;
            features[index++] = leftw_pdeprel;
            int pdeprel_leftpos = (pdeprel<<10) | leftpos;
            features[index++] = pdeprel_leftpos;
            int rightw_pdeprel = (rightw<<10) | pdeprel;
            features[index++] = rightw_pdeprel;
            int pdeprel_rightpos = (pdeprel<<10) | rightpos;
            features[index++] = pdeprel_rightpos;
            int leftsiblingw_pdeprel = (leftsiblingw<<10) |pdeprel;
            features[index++] = leftsiblingw_pdeprel;
            int pdeprel_leftsiblingpos = (pdeprel<<10) | leftsiblingpos;
            features[index++] = pdeprel_leftsiblingpos;
            int rightsiblingw_pdeprel = (rightsiblingw<<10) | pdeprel;
            features[index++] = rightsiblingw_pdeprel;
            int pdeprel_rightsiblingpos = (pdeprel<<10) | rightsiblingpos;
            features[index++] = pdeprel_rightsiblingpos;


            //plem + argument features
            long aw_plem = (aw<<20) | plem;
            features[index++] = aw_plem;
            int plem_apos = (plem<<10) | apos;
            features[index++] = plem_apos;
            int plem_adeprel = (plem<<10) | adeprel;
            features[index++] = plem_adeprel;
            //StringBuilder plem_deprelpath = new StringBuilder();
            //plem_deprelpath.append(plem);
            //plem_deprelpath.append(" ");
            //plem_deprelpath.append(deprelpath);
            String plem_deprelpath = plem +" "+deprelpath;
            features[index++] = plem_deprelpath;
            //StringBuilder plem_pospath= new StringBuilder();
            //plem_pospath.append(plem);
            //plem_pospath.append(" ");
            //plem_pospath.append(pospath);
            String plem_pospath = plem +" "+pospath;
            features[index++] = plem_pospath;
            int plem_position = (plem<<2) | position;
            features[index++] = plem_position;
            long leftw_plem = (leftw<<20) | plem;
            features[index++] = leftw_plem;
            int plem_leftpos = (plem<<10) | leftpos;
            features[index++] = plem_leftpos;
            long rightw_plem = (rightw<<20) | plem;
            features[index++] = rightw_plem;
            int plem_rightpos = (plem<<10) | rightpos;
            features[index++] = plem_rightpos;
            long leftsiblingw_plem = (leftsiblingw<<20) |plem;
            features[index++] = leftsiblingw_plem;
            int plem_leftsiblingpos = (plem<<10) | leftsiblingpos;
            features[index++] = plem_leftsiblingpos;
            long rightsiblingw_plem = (rightsiblingw<<20) | plem;
            features[index++] = rightsiblingw_plem;
            int plem_rightsiblingpos = (plem<<10) | rightsiblingpos;
            features[index++] = plem_rightsiblingpos;

            //psense + argument features
            //StringBuilder psense_aw = new StringBuilder();
            //psense_aw.append(psense);
            //psense_aw.append(" ");
            //psense_aw.append(aw);
            String psense_aw = pSense + " " + aw;
            features[index++] = psense_aw;
            //StringBuilder psense_apos = new StringBuilder();
            //psense_apos.append(psense);
            //psense_apos.append(" ");
            //psense_apos.append(apos);
            String psense_apos = pSense +" "+apos;
            features[index++] = psense_apos;
            //StringBuilder psense_adeprel = new StringBuilder();
            //psense_adeprel.append(psense);
            //psense_adeprel.append(" ");
            //psense_adeprel.append(adeprel);
            String psense_adeprel = pSense + " "+ adeprel;
            features[index++] = psense_adeprel;
            //StringBuilder psense_deprelpath = new StringBuilder();
            //psense_deprelpath.append(psense);
            //psense_deprelpath.append(" ");
            //psense_deprelpath.append(deprelpath);
            String psense_deprelpath = pSense + " "+ deprelpath;
            features[index++] = psense_deprelpath;
            //StringBuilder psense_pospath = new StringBuilder();
            //psense_pospath.append(psense);
            //psense_pospath.append(" ");
            //psense_pospath.append(pospath);
            String psense_pospath = pSense + " "+ pospath;
            features[index++] = psense_pospath;
            //StringBuilder psense_position = new StringBuilder();
            //psense_position.append(psense);
            //psense_position.append(" ");
            //psense_position.append(position);
            String psense_position = pSense + " " +position;
            features[index++] = psense_position;
            //StringBuilder psense_leftw = new StringBuilder();
            //psense_leftw.append(psense);
            //psense_leftw.append(" ");
            //psense_leftw.append(leftw);
            String psense_leftw = pSense + " " + leftw;
            features[index++] = psense_leftw;
            //StringBuilder psense_leftpos = new StringBuilder();
            //psense_leftpos.append(psense);
            //psense_leftpos.append(" ");
            //psense_leftpos.append(leftpos);
            String psense_leftpos = pSense + " " + leftpos;
            features[index++] = psense_leftpos;
            //StringBuilder psense_rightw = new StringBuilder();
            //psense_rightw.append(psense);
            //psense_rightw.append(" ");
            //psense_rightw.append(rightw);
            String psense_rightw = pSense + " " + rightw;
            features[index++] = psense_rightw;
            //StringBuilder psense_rightpos = new StringBuilder();
            //psense_rightpos.append(psense);
            //psense_rightpos.append(" ");
            //psense_rightpos.append(rightpos);
            String psense_rightpos = pSense + " " + rightpos;
            features[index++] = psense_rightpos;
            //StringBuilder psense_leftsiblingw = new StringBuilder();
            //psense_leftsiblingw.append(psense);
            //psense_leftsiblingw.append(" ");
            //psense_leftsiblingw.append(leftsiblingw);
            String psense_leftsiblingw = pSense +" " + leftsiblingw;
            features[index++] = psense_leftsiblingw;
            //StringBuilder psense_leftsiblingpos = new StringBuilder();
            //psense_leftsiblingpos.append(psense);
            //psense_leftsiblingpos.append(" ");
            //psense_leftsiblingpos.append(leftsiblingpos);
            String psense_leftsiblingpos = pSense + " " + leftsiblingpos;
            features[index++] = psense_leftsiblingpos;
            //StringBuilder psense_rightsiblingw = new StringBuilder();
            //psense_rightsiblingw.append(psense);
            //psense_rightsiblingw.append(" ");
            //psense_rightsiblingw.append(rightsiblingw);
            String psense_rightsiblingw = pSense +" "+rightsiblingw;
            features[index++] = psense_rightsiblingw;
            //StringBuilder psense_rightsiblingpos = new StringBuilder();
            //psense_rightsiblingpos.append(psense);
            //psense_rightsiblingpos.append(psense);
            //psense_rightsiblingpos.append(" ");
            //psense_rightsiblingpos.append(rightsiblingpos);
            String psense_rightsiblingpos = pSense +" "+rightsiblingpos;
            features[index++] = psense_rightsiblingpos;


            //pprw  + argument features
            long aw_pprw = (aw<<20) | pprw;
            features[index++] = aw_pprw;
            int pprw_apos = (pprw<<10) | apos;
            features[index++] = pprw_apos;
            int pprw_adeprel = (pprw<<10) | adeprel;
            features[index++] = pprw_adeprel;
            //StringBuilder pprw_deprelpath = new StringBuilder();
            //pprw_deprelpath.append(pprw);
            //pprw_deprelpath.append(" ");
            //pprw_deprelpath.append(deprelpath);
            String pprw_deprelpath = pprw +" "+deprelpath;
            features[index++] = pprw_deprelpath;
            //StringBuilder pprw_pospath= new StringBuilder();
            //pprw_pospath.append(pprw);
            //pprw_pospath.append(" ");
            //pprw_pospath.append(pospath);
            String pprw_pospath = pprw +" "+ pospath;
            features[index++] = pprw_pospath;
            int pprw_position = (pprw<<2) | position;
            features[index++] = pprw_position;
            long leftw_pprw = (leftw<<20) | pprw;
            features[index++] = leftw_pprw;
            int pprw_leftpos = (pprw<<10) | leftpos;
            features[index++] = pprw_leftpos;
            long rightw_pprw = (rightw<<20) | pprw;
            features[index++] = rightw_pprw;
            int pprw_rightpos = (pprw<<10) | rightpos;
            features[index++] = pprw_rightpos;
            long leftsiblingw_pprw = (leftsiblingw<<20) |pprw;
            features[index++] = leftsiblingw_pprw;
            int pprw_leftsiblingpos = (pprw<<10) | leftsiblingpos;
            features[index++] = pprw_leftsiblingpos;
            long rightsiblingw_pprw = (rightsiblingw<<20) | pprw;
            features[index++] = rightsiblingw_pprw;
            int pprw_rightsiblingpos = (pprw<<10) | rightsiblingpos;
            features[index++] = pprw_rightsiblingpos;


            //pdeprel + argument features
            int aw_pprpos = (aw<<10) | pprpos;
            features[index++] = aw_pprpos;
            int pprpos_apos = (pprpos<<10) | apos;
            features[index++] = pprpos_apos;
            int pprpos_adeprel = (pprpos<<10) | adeprel;
            features[index++] = pprpos_adeprel;
            //StringBuilder pprpos_deprelpath = new StringBuilder();
            //pprpos_deprelpath.append(pprpos);
            //pprpos_deprelpath.append(" ");
            //pprpos_deprelpath.append(deprelpath);
            String pprpos_deprelpath = pprpos +" "+deprelpath;
            features[index++] = pprpos_deprelpath;
            //StringBuilder pprpos_pospath= new StringBuilder();
            //pprpos_pospath.append(pprpos);
            //pprpos_pospath.append(" ");
            //pprpos_pospath.append(pospath);
            String pprpos_pospath= pprpos +" "+pospath;
            features[index++] = pprpos_pospath;
            int pprpos_position = (pprpos<<2) | position;
            features[index++] = pprpos_position;
            int leftw_pprpos = (leftw<<10) | pprpos;
            features[index++] = leftw_pprpos;
            int pprpos_leftpos = (pprpos<<10) | leftpos;
            features[index++] = pprpos_leftpos;
            int rightw_pprpos = (rightw<<10) | pprpos;
            features[index++] = rightw_pprpos;
            int pprpos_rightpos = (pprpos<<10) | rightpos;
            features[index++] = pprpos_rightpos;
            int leftsiblingw_pprpos = (leftsiblingw<<10) |pprpos;
            features[index++] = leftsiblingw_pprpos;
            int pprpos_leftsiblingpos = (pprpos<<10) | leftsiblingpos;
            features[index++] = pprpos_leftsiblingpos;
            int rightsiblingw_pprpos = (rightsiblingw<<10) | pprpos;
            features[index++] = rightsiblingw_pprpos;
            int pprpos_rightsiblingpos = (pprpos<<10) | rightsiblingpos;
            features[index++] = pprpos_rightsiblingpos;

            //pchilddepset + argument features
            //StringBuilder pchilddepset_aw = new StringBuilder();
            //pchilddepset_aw.append(pchilddepset);
            //pchilddepset_aw.append(" ");
            //pchilddepset_aw.append(aw);
            String pchilddepset_aw = pchilddepset +" "+aw;
            features[index++] = pchilddepset_aw;
            //StringBuilder pchilddepset_apos = new StringBuilder();
            //pchilddepset_apos.append(pchilddepset);
            //pchilddepset_apos.append(" ");
            //pchilddepset_apos.append(apos);
            String pchilddepset_apos = pchilddepset +" "+ apos;
            features[index++] = pchilddepset_apos;
            //StringBuilder pchilddepset_adeprel = new StringBuilder();
            //pchilddepset_adeprel.append(pchilddepset);
            //pchilddepset_adeprel.append(" ");
            //pchilddepset_adeprel.append(adeprel);
            String pchilddepset_adeprel = pchilddepset +" "+ adeprel;
            features[index++] = pchilddepset_adeprel;
            //StringBuilder pchilddepset_deprelpath = new StringBuilder();
            //pchilddepset_deprelpath.append(pchilddepset);
            //pchilddepset_deprelpath.append(" ");
            //pchilddepset_deprelpath.append(deprelpath);
            String pchilddepset_deprelpath = pchilddepset +" "+ deprelpath;
            features[index++] = pchilddepset_deprelpath;
            //StringBuilder pchilddepset_pospath = new StringBuilder();
            //pchilddepset_pospath.append(pchilddepset);
            //pchilddepset_pospath.append(" ");
            //pchilddepset_pospath.append(pospath);
            String pchilddepset_pospath = pchilddepset +" "+pospath;
            features[index++] = pchilddepset_pospath;
            //StringBuilder pchilddepset_position = new StringBuilder();
            //pchilddepset_position.append(pchilddepset);
            //pchilddepset_position.append(" ");
            //pchilddepset_position.append(position);
            String pchilddepset_position = pchilddepset +" "+position;
            features[index++] = pchilddepset_position;
            //StringBuilder pchilddepset_leftw = new StringBuilder();
            //pchilddepset_leftw.append(pchilddepset);
            //pchilddepset_leftw.append(" ");
            //pchilddepset_leftw.append(leftw);
            String pchilddepset_leftw = pchilddepset +" "+ leftw;
            features[index++] = pchilddepset_leftw;
            //StringBuilder pchilddepset_leftpos = new StringBuilder();
            //pchilddepset_leftpos.append(pchilddepset);
            //pchilddepset_leftpos.append(" ");
            //pchilddepset_leftpos.append(leftpos);
            String pchilddepset_leftpos = pchilddepset +" "+leftpos;
            features[index++] = pchilddepset_leftpos;
            //StringBuilder pchilddepset_rightw = new StringBuilder();
            //pchilddepset_rightw.append(pchilddepset);
            //pchilddepset_rightw.append(" ");
            //pchilddepset_rightw.append(rightw);
            String pchilddepset_rightw = pchilddepset +" "+rightw;
            features[index++] = pchilddepset_rightw;
            //StringBuilder pchilddepset_rightpos = new StringBuilder();
            //pchilddepset_rightpos.append(pchilddepset);
            //pchilddepset_rightpos.append(" ");
            //pchilddepset_rightpos.append(rightpos);
            String pchilddepset_rightpos = pchilddepset +" "+rightpos;
            features[index++] = pchilddepset_rightpos;
            //StringBuilder pchilddepset_leftsiblingw = new StringBuilder();
            //pchilddepset_leftsiblingw.append(pchilddepset);
            //pchilddepset_leftsiblingw.append(" ");
            //pchilddepset_leftsiblingw.append(leftsiblingw);
            String pchilddepset_leftsiblingw = pchilddepset +" "+leftsiblingw;
            features[index++] = pchilddepset_leftsiblingw;
            //StringBuilder pchilddepset_leftsiblingpos = new StringBuilder();
            //pchilddepset_leftsiblingpos.append(pchilddepset);
            //pchilddepset_leftsiblingpos.append(" ");
            //pchilddepset_leftsiblingpos.append(leftsiblingpos);
            String pchilddepset_leftsiblingpos = pchilddepset +" "+ leftsiblingpos;
            features[index++] = pchilddepset_leftsiblingpos;
            //StringBuilder pchilddepset_rightsiblingw = new StringBuilder();
            //pchilddepset_rightsiblingw.append(pchilddepset);
            //pchilddepset_rightsiblingw.append(" ");
            //pchilddepset_rightsiblingw.append(rightsiblingw);
            String pchilddepset_rightsiblingw = pchilddepset +" "+rightsiblingw;
            features[index++] = pchilddepset_rightsiblingw;
            //StringBuilder pchilddepset_rightsiblingpos = new StringBuilder();
            //pchilddepset_rightsiblingpos.append(pchilddepset);
            //pchilddepset_rightsiblingpos.append(" ");
            //pchilddepset_rightsiblingpos.append(rightsiblingpos);
            String pchilddepset_rightsiblingpos = pchilddepset + " "+rightsiblingpos;
            features[index++] = pchilddepset_rightsiblingpos;


            //pdepsubcat + argument features
            //StringBuilder pdepsubcat_aw = new StringBuilder();
            //pdepsubcat_aw.append(pdepsubcat);
            //pdepsubcat_aw.append(" ");
            //pdepsubcat_aw.append(aw);
            String pdepsubcat_aw =pdepsubcat +" " + aw;
            features[index++] = pdepsubcat_aw;
            //StringBuilder pdepsubcat_apos = new StringBuilder();
            //pdepsubcat_apos.append(pdepsubcat);
            //pdepsubcat_apos.append(" ");
            //pdepsubcat_apos.append(apos);
            String pdepsubcat_apos = pdepsubcat +" " + apos;
            features[index++] = pdepsubcat_apos;
            //StringBuilder pdepsubcat_adeprel = new StringBuilder();
            //pdepsubcat_adeprel.append(pdepsubcat);
            //pdepsubcat_adeprel.append(" ");
            //pdepsubcat_adeprel.append(adeprel);
            String pdepsubcat_adeprel = pdepsubcat +" "+adeprel;
            features[index++] = pdepsubcat_adeprel;
            //StringBuilder pdepsubcat_deprelpath = new StringBuilder();
            //pdepsubcat_deprelpath.append(pdepsubcat);
            //pdepsubcat_deprelpath.append(" ");
            //pdepsubcat_deprelpath.append(deprelpath);
            String pdepsubcat_deprelpath = pdepsubcat +" "+deprelpath;
            features[index++] = pdepsubcat_deprelpath;
            //StringBuilder pdepsubcat_pospath = new StringBuilder();
            //pdepsubcat_pospath.append(pdepsubcat);
            //pdepsubcat_pospath.append(" ");
            //pdepsubcat_pospath.append(pospath);
            String pdepsubcat_pospath = pdepsubcat +" "+pospath;
            features[index++] = pdepsubcat_pospath;
            //StringBuilder pdepsubcat_position = new StringBuilder();
            //pdepsubcat_position.append(pdepsubcat);
            //pdepsubcat_position.append(" ");
            //pdepsubcat_position.append(position);
            String pdepsubcat_position = pdepsubcat +" "+position;
            features[index++] = pdepsubcat_position;
            //StringBuilder pdepsubcat_leftw = new StringBuilder();
            //pdepsubcat_leftw.append(pdepsubcat);
            //pdepsubcat_leftw.append(" ");
            //pdepsubcat_leftw.append(leftw);
            String pdepsubcat_leftw = pdepsubcat +" "+leftw;
            features[index++] = pdepsubcat_leftw;
            //StringBuilder pdepsubcat_leftpos = new StringBuilder();
            //pdepsubcat_leftpos.append(pdepsubcat);
            //pdepsubcat_leftpos.append(" ");
            //pdepsubcat_leftpos.append(leftpos);
            String pdepsubcat_leftpos =pdepsubcat +" "+ leftpos;
            features[index++] = pdepsubcat_leftpos;
            //StringBuilder pdepsubcat_rightw = new StringBuilder();
            //pdepsubcat_rightw.append(pdepsubcat);
            //pdepsubcat_rightw.append(" ");
            //pdepsubcat_rightw.append(rightw);
            String pdepsubcat_rightw = pdepsubcat +" "+ rightw;
            features[index++] = pdepsubcat_rightw;
            //StringBuilder pdepsubcat_rightpos = new StringBuilder();
            //pdepsubcat_rightpos.append(pdepsubcat);
            //pdepsubcat_rightpos.append(" ");
            //pdepsubcat_rightpos.append(rightpos);
            String pdepsubcat_rightpos = pdepsubcat +" "+rightpos;
            features[index++] = pdepsubcat_rightpos;
            //StringBuilder pdepsubcat_leftsiblingw = new StringBuilder();
            //pdepsubcat_leftsiblingw.append(pdepsubcat);
            //pdepsubcat_leftsiblingw.append(" ");
            //pdepsubcat_leftsiblingw.append(leftsiblingw);
            String pdepsubcat_leftsiblingw =pdepsubcat +" "+ leftsiblingw;
            features[index++] = pdepsubcat_leftsiblingw;
            //StringBuilder pdepsubcat_leftsiblingpos = new StringBuilder();
            //pdepsubcat_leftsiblingpos.append(pdepsubcat);
            //pdepsubcat_leftsiblingpos.append(" ");
            //pdepsubcat_leftsiblingpos.append(leftsiblingpos);
            String pdepsubcat_leftsiblingpos = pdepsubcat + " "+leftsiblingpos;
            features[index++] = pdepsubcat_leftsiblingpos;
            //StringBuilder pdepsubcat_rightsiblingw = new StringBuilder();
            //pdepsubcat_rightsiblingw.append(pdepsubcat);
            //pdepsubcat_rightsiblingw.append(" ");
            //pdepsubcat_rightsiblingw.append(rightsiblingw);
            String pdepsubcat_rightsiblingw = pdepsubcat +" "+rightsiblingw;
            features[index++] = pdepsubcat_rightsiblingw;
            //StringBuilder pdepsubcat_rightsiblingpos = new StringBuilder();
            //pdepsubcat_rightsiblingpos.append(pdepsubcat);
            //pdepsubcat_rightsiblingpos.append(" ");
            //pdepsubcat_rightsiblingpos.append(rightsiblingpos);
            String pdepsubcat_rightsiblingpos = pdepsubcat +" "+rightsiblingpos;
            features[index++] = pdepsubcat_rightsiblingpos;


            //pchildposset + argument features
            //StringBuilder pchildposset_aw = new StringBuilder();
            ///pchildposset_aw.append(pchildposset);
            //pchildposset_aw.append(" ");
            //pchildposset_aw.append(aw);
            String pchildposset_aw = pchildposset + " "+ aw;
            features[index++] = pchildposset_aw;
            //StringBuilder pchildposset_apos = new StringBuilder();
            //pchildposset_apos.append(pchildposset);
            //pchildposset_apos.append(" ");
            //pchildposset_apos.append(apos);
            String pchildposset_apos = pchildposset +" "+apos;
            features[index++] = pchildposset_apos;
            //StringBuilder pchildposset_adeprel = new StringBuilder();
            //pchildposset_adeprel.append(pchildposset);
            //pchildposset_adeprel.append(" ");
            //pchildposset_adeprel.append(adeprel);
            String pchildposset_adeprel = pchildposset + " " + adeprel;
            features[index++] = pchildposset_adeprel;
            //StringBuilder pchildposset_deprelpath = new StringBuilder();
            //pchildposset_deprelpath.append(pchildposset);
            //pchildposset_deprelpath.append(" ");
            //pchildposset_deprelpath.append(deprelpath);
            String pchildposset_deprelpath =pchildposset +" "+deprelpath;
            features[index++] = pchildposset_deprelpath;
            //StringBuilder pchildposset_pospath = new StringBuilder();
            //pchildposset_pospath.append(pchildposset);
            //pchildposset_pospath.append(" ");
            //pchildposset_pospath.append(pospath);
            String pchildposset_pospath = pchildposset +" "+ pospath;
            features[index++] = pchildposset_pospath;
            //StringBuilder pchildposset_position = new StringBuilder();
            //pchildposset_position.append(pchildposset);
            //pchildposset_position.append(" ");
            //pchildposset_position.append(position);
            String pchildposset_position = pchildposset + " "+ position;
            features[index++] = pchildposset_position;
            //StringBuilder pchildposset_leftw = new StringBuilder();
            //pchildposset_leftw.append(pchildposset);
            //pchildposset_leftw.append(" ");
            //pchildposset_leftw.append(leftw);
            String pchildposset_leftw = pchildposset +" "+leftw;
            features[index++] = pchildposset_leftw;
            //StringBuilder pchildposset_leftpos = new StringBuilder();
            //pchildposset_leftpos.append(pchildposset);
            //pchildposset_leftpos.append(" ");
            //pchildposset_leftpos.append(leftpos);
            String pchildposset_leftpos = pchildposset +" "+ leftpos;
            features[index++] = pchildposset_leftpos;
            //StringBuilder pchildposset_rightw = new StringBuilder();
            //pchildposset_rightw.append(pchildposset);
            //pchildposset_rightw.append(" ");
            //pchildposset_rightw.append(rightw);
            String pchildposset_rightw = pchildposset + " "+rightw;
            features[index++] = pchildposset_rightw;
            //StringBuilder pchildposset_rightpos = new StringBuilder();
            //pchildposset_rightpos.append(pchildposset);
            //pchildposset_rightpos.append(" ");
            //pchildposset_rightpos.append(rightpos);
            String pchildposset_rightpos = pchildposset + " "+rightpos;
            features[index++] = pchildposset_rightpos;
            //StringBuilder pchildposset_leftsiblingw = new StringBuilder();
            //pchildposset_leftsiblingw.append(pchildposset);
            //pchildposset_leftsiblingw.append(" ");
            //pchildposset_leftsiblingw.append(leftsiblingw);
            String pchildposset_leftsiblingw = pchildposset + " "+ leftsiblingw;
            features[index++] = pchildposset_leftsiblingw;
            //StringBuilder pchildposset_leftsiblingpos = new StringBuilder();
            //pchildposset_leftsiblingpos.append(pchildposset);
            //pchildposset_leftsiblingpos.append(" ");
            //pchildposset_leftsiblingpos.append(leftsiblingpos);
            String pchildposset_leftsiblingpos = pchildposset + " "+ leftsiblingpos;
            features[index++] = pchildposset_leftsiblingpos;
            //StringBuilder pchildposset_rightsiblingw = new StringBuilder();
            //pchildposset_rightsiblingw.append(pchildposset);
            //pchildposset_rightsiblingw.append(" ");
            //pchildposset_rightsiblingw.append(rightsiblingw);
            String pchildposset_rightsiblingw = pchildposset +" " +rightsiblingw;
            features[index++] = pchildposset_rightsiblingw;
            //StringBuilder pchildposset_rightsiblingpos = new StringBuilder();
            //pchildposset_rightsiblingpos.append(pchildposset);
            //pchildposset_rightsiblingpos.append(" ");
            //pchildposset_rightsiblingpos.append(rightsiblingpos);
            String pchildposset_rightsiblingpos = pchildposset +" "+rightsiblingpos;
            features[index++] = pchildposset_rightsiblingpos;


            //pchildwset + argument features
            //StringBuilder pchildwset_aw = new StringBuilder();
            //pchildwset_aw.append(pchildwset);
            //pchildwset_aw.append(" ");
            //pchildwset_aw.append(aw);
            String pchildwset_aw = pchildwset +" "+ aw;
            features[index++] = pchildwset_aw;
            //StringBuilder pchildwset_apos = new StringBuilder();
            //pchildwset_apos.append(pchildwset);
            //pchildwset_apos.append(" ");
            //pchildwset_apos.append(apos);
            String pchildwset_apos = pchildwset + " "+apos;
            features[index++] = pchildwset_apos;
            //StringBuilder pchildwset_adeprel = new StringBuilder();
            //pchildwset_adeprel.append(pchildwset);
            //pchildwset_adeprel.append(" ");
            //pchildwset_adeprel.append(adeprel);
            String pchildwset_adeprel = pchildwset +" "+adeprel;
            features[index++] = pchildwset_adeprel;
            //StringBuilder pchildwset_deprelpath = new StringBuilder();
            //pchildwset_deprelpath.append(pchildwset);
            //pchildwset_deprelpath.append(" ");
            //pchildwset_deprelpath.append(deprelpath);
            String pchildwset_deprelpath = pchildwset +" "+deprelpath;
            features[index++] = pchildwset_deprelpath;
            //StringBuilder pchildwset_pospath = new StringBuilder();
            //pchildwset_pospath.append(pchildwset);
            //pchildwset_pospath.append(" ");
            //pchildwset_pospath.append(pospath);
            String pchildwset_pospath = pchildwset +" "+pospath;
            features[index++] = pchildwset_pospath;
            //StringBuilder pchildwset_position = new StringBuilder();
            //pchildwset_position.append(pchildwset);
            //pchildwset_position.append(" ");
            //pchildwset_position.append(position);
            String pchildwset_position = pchildwset +" "+ position;
            features[index++] = pchildwset_position;
            //StringBuilder pchildwset_leftw = new StringBuilder();
            //pchildwset_leftw.append(pchildwset);
            //pchildwset_leftw.append(" ");
            //pchildwset_leftw.append(leftw);
            String pchildwset_leftw = pchildwset +" "+leftw;
            features[index++] = pchildwset_leftw;
            //StringBuilder pchildwset_leftpos = new StringBuilder();
            //pchildwset_leftpos.append(pchildwset);
            //pchildwset_leftpos.append(" ");
            //pchildwset_leftpos.append(leftpos);
            String pchildwset_leftpos = pchildwset +" "+leftpos;
            features[index++] = pchildwset_leftpos;
            //StringBuilder pchildwset_rightw = new StringBuilder();
            //pchildwset_rightw.append(pchildwset);
            //pchildwset_rightw.append(" ");
            //pchildwset_rightw.append(rightw);
            String pchildwset_rightw = pchildwset +" "+rightw;
            features[index++] = pchildwset_rightw;
            //StringBuilder pchildwset_rightpos = new StringBuilder();
            //pchildwset_rightpos.append(pchildwset);
            //pchildwset_rightpos.append(" ");
            //pchildwset_rightpos.append(rightpos);
            String pchildwset_rightpos = pchildwset + " "+ rightpos;
            features[index++] = pchildwset_rightpos;
            //StringBuilder pchildwset_leftsiblingw = new StringBuilder();
            //pchildwset_leftsiblingw.append(pchildwset);
            //pchildwset_leftsiblingw.append(" ");
            //pchildwset_leftsiblingw.append(leftsiblingw);
            String pchildwset_leftsiblingw = pchildwset +" "+leftsiblingw;
            features[index++] = pchildwset_leftsiblingw;
            //StringBuilder pchildwset_leftsiblingpos = new StringBuilder();
            //pchildwset_leftsiblingpos.append(pchildwset);
            //pchildwset_leftsiblingpos.append(" ");
            //pchildwset_leftsiblingpos.append(leftsiblingpos);
            String pchildwset_leftsiblingpos = pchildwset +" "+leftsiblingpos;
            features[index++] = pchildwset_leftsiblingpos;
            //StringBuilder pchildwset_rightsiblingw = new StringBuilder();
            //pchildwset_rightsiblingw.append(pchildwset);
            //pchildwset_rightsiblingw.append(" ");
            //pchildwset_rightsiblingw.append(rightsiblingw);
            String pchildwset_rightsiblingw = pchildwset +" " + rightsiblingw;
            features[index++] = pchildwset_rightsiblingw;
            //StringBuilder pchildwset_rightsiblingpos = new StringBuilder();
            //pchildwset_rightsiblingpos.append(pchildwset);
            //pchildwset_rightsiblingpos.append(" ");
            //pchildwset_rightsiblingpos.append(rightsiblingpos);
            String pchildwset_rightsiblingpos = pchildwset +" "+rightsiblingpos;
            features[index++] = pchildwset_rightsiblingpos;
            */
        }
        else if (state.equals("AC"))
        {
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
            ///  PREDICATE-PREDICATE CONJOINED FEATURES BASED ON ORIGINAL PAPER //
            //////////////////////////////////////////////////////////////////////
            /*
            String psense_aw = pSense + " " + aw;
            features[index++] = psense_aw;
            String psense_apos = pSense +" "+apos;
            features[index++] = psense_apos;
            String psense_pospath = pSense + " "+ pospath;
            features[index++] = psense_pospath;
            int aw_position = (aw<<2) | position;
            features[index++] = aw_position;
            String psense_position = pSense + " " +position;
            features[index++] = psense_position;
            String psense_rightsiblingpos = pSense +" "+rightsiblingpos;
            features[index++] = psense_rightsiblingpos;
            int leftpos_rightsiblingpos = (leftpos<<10) | rightsiblingpos;
            features[index++] = leftpos_rightsiblingpos;
            int apos_position = (apos<<2) | position;
            features[index++] = apos_position;
            String psense_leftsiblingpos = pSense + " " + leftsiblingpos;
            features[index++] = psense_leftsiblingpos;
            String psense_adeprel = pSense + " "+ adeprel;
            features[index++] = psense_adeprel;
            String pchilddepset_position = pchilddepset +" "+position;
            features[index++] = pchilddepset_position;
            int adeprel_rightpos = (adeprel<<10) | rightpos;
            features[index++] = adeprel_rightpos;
            int aw_apos = (aw<<10) | apos;
            features[index++] = aw_apos;
            int plem_apos = (plem<<10) | apos;
            features[index++] = plem_apos;
            int apos_adeprel = (apos<<10) | adeprel;
            features[index++] = apos_adeprel;
            */

            //////////////////////////////////////////////////////////////////////
            ///////////////// PREDICATE-PREDICATE CONJOINED FEATURES (55) ////////
            //////////////////////////////////////////////////////////////////////

            //todo 20 bits for words, 10 bits for pos, 10 bits for dependencies
            // pw, ppos, plem, pdeprel, pSense, pprw, pprpos, pdepsubcat, pchilddepset, pchildposset, pchildwset;
            /*
            //pw + ... (20 bits for pw)
            int pw_ppos = (pw<<10) | ppos ;
            features[index++] = pw_ppos;
            long pw_plem = (pw<<20) | plem;
            features[index++] = pw_plem;
            int pw_pdeprel = (pw<<10) | pdeprel;
            features[index++] = pw_pdeprel;
            String pw_psense = pw + " " + pSense;
            features[index++] = pw_psense;
            long pw_pprw = (pw<<20) | pprw;
            features[index++] = pw_pprw;
            int pw_pprpos = (pw<<10) | pprpos ;
            features[index++] = pw_pprpos;
            String pw_pdepsubcat = pw + " " + pdepsubcat;
            features[index++] = pw_pdepsubcat;
            String pw_pchilddepset = pw + " " + pchilddepset;
            features[index++] = pw_pchilddepset;
            String pw_pchildposset = pw + " " + pchildposset;
            features[index++] = pw_pchildposset;
            String pw_pchildwset = pw + " " + pchildwset;
            features[index++] = pw_pchildwset;

            //ppos + ...(10 bits for ppos)
            int ppos_plem = (plem<<10) | ppos;
            features[index++] = ppos_plem;
            int ppos_pdeprel = (ppos<<10) | pdeprel;
            features[index++] = ppos_pdeprel;
            String ppos_psense = ppos + " " + pSense;
            features[index++] = ppos_psense;
            int ppos_pprw = (pprw<<10) | ppos;
            features[index++] = ppos_pprw;
            int ppos_pprpos = (ppos<<10) | pprpos ;
            features[index++] = ppos_pprpos;
            String ppos_pdepsubcat = ppos + " " + pdepsubcat;
            features[index++] = ppos_pdepsubcat;
            String ppos_pchilddepset = ppos + " " + pchilddepset;
            features[index++] = ppos_pchilddepset;
            String ppos_pchildposset = ppos + " " + pchildposset;
            features[index++] = ppos_pchildposset;
            String ppos_pchildwset = ppos + " " + pchildwset;
            features[index++] = ppos_pchildwset;

            //plem + ... (20 bits for plem)
            int plem_pdeprel = (plem<<10) | pdeprel;
            features[index++] = plem_pdeprel;
            String plem_psense = plem + " " + pSense;
            features[index++] = plem_psense;
            long plem_pprw = (plem<<20) | pprw;
            features[index++] = plem_pprw;
            int plem_pprpos = (plem<<10) | pprpos ;
            features[index++] = plem_pprpos;
            String plem_pdepsubcat = plem + " " + pdepsubcat;
            features[index++] = plem_pdepsubcat;
            String plem_pchilddepset = plem + " " + pchilddepset;
            features[index++] = plem_pchilddepset;
            String plem_pchildposset = plem + " " + pchildposset;
            features[index++] = plem_pchildposset;
            String plem_pchildwset = plem + " " + pchildwset;
            features[index++] = plem_pchildwset;

            //pdeprel + ...(10 bits for pdeprel)
            String pdeprel_psense = pdeprel + " " + pSense;
            features[index++] = pdeprel_psense;
            int pdeprel_pprw = (pprw<<10) | pdeprel;
            features[index++] = pdeprel_pprw;
            int pdeprel_pprpos = (pdeprel<<10) | pprpos ;
            features[index++] = pdeprel_pprpos;
            String pdeprel_pdepsubcat = pdeprel + " " + pdepsubcat;
            features[index++] = pdeprel_pdepsubcat;
            String pdeprel_pchilddepset = pdeprel + " " + pchilddepset;
            features[index++] = pdeprel_pchilddepset;
            String pdeprel_pchildposset = pdeprel + " " + pchildposset;
            features[index++] = pdeprel_pchildposset;
            String pdeprel_pchildwset = pdeprel + " " + pchildwset;
            features[index++] = pdeprel_pchildwset;

            //psense + ...
            String psense_pprw = pSense+ " "+ pprw;
            features[index++] = psense_pprw;
            String psense_pprpos = pSense+" "+ pprpos ;
            features[index++] = psense_pprpos;
            String psense_pdepsubcat = pSense + " " + pdepsubcat;
            features[index++] = psense_pdepsubcat;
            String psense_pchilddepset = pSense + " " + pchilddepset;
            features[index++] = psense_pchilddepset;
            String psense_pchildposset = pSense + " " + pchildposset;
            features[index++] = psense_pchildposset;
            String psense_pchildwset = pSense + " " + pchildwset;
            features[index++] = psense_pchildwset;

            //pprw + ... (20 bits for pprw)
            int pprw_pprpos = (pprw<<10) | pprpos ;
            features[index++] = pprw_pprpos;
            String pprw_pdepsubcat = pprw + " " + pdepsubcat;
            features[index++] = pprw_pdepsubcat;
            String pprw_pchilddepset = pprw + " " + pchilddepset;
            features[index++] = pprw_pchilddepset;
            String pprw_pchildposset = pprw + " " + pchildposset;
            features[index++] = pprw_pchildposset;
            String pprw_pchildwset = pprw + " " + pchildwset;
            features[index++] = pprw_pchildwset;

            //pprpos + ...(10 bits for pprpos)
            String pprpos_pdepsubcat = pprpos + " " + pdepsubcat;
            features[index++] = pprpos_pdepsubcat;
            String pprpos_pchilddepset = pprpos + " " + pchilddepset;
            features[index++] = pprpos_pchilddepset;
            String pprpos_pchildposset = pprpos + " " + pchildposset;
            features[index++] = pprpos_pchildposset;
            String pprpos_pchildwset = pprpos + " " + pchildwset;
            features[index++] = pprpos_pchildwset;

            //pdepsubcat + ...
            String pdepsubcat_pchilddepset = pdepsubcat + " " + pchilddepset;
            features[index++] = pdepsubcat_pchilddepset;
            String pdepsubcat_pchildposset = pdepsubcat + " " + pchildposset;
            features[index++] = pdepsubcat_pchildposset;
            String pdepsubcat_pchildwset = pdepsubcat + " " + pchildwset;
            features[index++] = pdepsubcat_pchildwset;

            //pchilddepset + ...
            String pchilddepset_pchildposset = pchilddepset + " " + pchildposset;
            features[index++] = pchilddepset_pchildposset;
            String pchilddepset_pchildwset = pchilddepset + " " + pchildwset;
            features[index++] = pchilddepset_pchildwset;

            //pchildposset + ...
            String pchildposset_pchildwset = pchildposset + " " + pchildwset;
            features[index++] = pchildposset_pchildwset;

            //////////////////////////////////////////////////////////////////////
            ///////////////// ARGUMENT-ARGUMENT CONJOINED FEATURES (91) /////////
            //////////////////////////////////////////////////////////////////////
            //todo 20 bits for words, 10 bits for pos, 10 bits for dependencies
            // aw, apos, adeprel, deprelpath, pospath, position, leftw, leftpos, rightw, rightpos, leftsiblingw,
            // leftsiblingpos, rightsiblingw, rightsiblingpos

            int aw_apos = (aw<<10) | apos;
            features[index++] = aw_apos;
            int aw_adeprel = (aw<<10) | adeprel;
            features[index++] = aw_adeprel;
            String aw_deprelpath = aw+" "+ deprelpath;
            features[index++] = aw_deprelpath;
            String aw_pospath = aw+" "+ pospath;
            features[index++] = aw_pospath;
            int aw_position = (aw<<2) | position;
            features[index++] = aw_position;
            long aw_leftw = (aw<< 20) | leftw;
            features[index++] = aw_leftw;
            int aw_leftpos = (aw<<10) | leftpos;
            features[index++] = aw_leftpos;
            long aw_rightw = (aw<< 20) | rightw;
            features[index++] = aw_rightw;
            int aw_rightpos = (aw<<10) | rightpos;
            features[index++] = aw_rightpos;
            long aw_leftsiblingw = (aw<< 20) | leftsiblingw;
            features[index++] = aw_leftsiblingw;
            int aw_leftsiblingpos = (aw<<10) | leftsiblingpos;
            features[index++] = aw_leftsiblingpos;
            long aw_rightsiblingw = (aw<< 20) | rightsiblingw;
            features[index++] = aw_rightsiblingw;
            int aw_rightsiblingpos = (aw<<10) | rightsiblingpos;
            features[index++] = aw_rightsiblingpos;

            int apos_adeprel = (apos<<10) | adeprel;
            features[index++] = apos_adeprel;
            String apos_deprelpath = apos+" "+ deprelpath;
            features[index++] = apos_deprelpath;
            String apos_pospath = apos+" "+ pospath;
            features[index++] = apos_pospath;
            int apos_position = (apos<<2) | position;
            features[index++] = apos_position;
            int apos_leftw = (leftw<< 10) | apos;
            features[index++] = apos_leftw;
            int apos_leftpos = (apos<<10) | leftpos;
            features[index++] = apos_leftpos;
            int apos_rightw = (rightw<< 10) | apos;
            features[index++] = apos_rightw;
            int apos_rightpos = (apos<<10) | rightpos;
            features[index++] = apos_rightpos;
            int apos_leftsiblingw = (leftsiblingw<< 10) | apos;
            features[index++] = apos_leftsiblingw;
            int apos_leftsiblingpos = (apos<<10) | leftsiblingpos;
            features[index++] = apos_leftsiblingpos;
            int apos_rightsiblingw = (rightsiblingw<< 10) | apos;
            features[index++] = apos_rightsiblingw;
            int apos_rightsiblingpos = (apos<<10) | rightsiblingpos;
            features[index++] = apos_rightsiblingpos;

            String adeprel_deprelpath = adeprel+" "+ deprelpath;
            features[index++] = adeprel_deprelpath;
            String adeprel_pospath = adeprel+" "+ pospath;
            features[index++] = adeprel_pospath;
            int adeprel_position = (adeprel<<2) | position;
            features[index++] = adeprel_position;
            int adeprel_leftw = (leftw<< 10) | adeprel;
            features[index++] = adeprel_leftw;
            int adeprel_leftpos = (adeprel<<10) | leftpos;
            features[index++] = adeprel_leftpos;
            int adeprel_rightw = (rightw<< 10) | adeprel;
            features[index++] = adeprel_rightw;
            int adeprel_rightpos = (adeprel<<10) | rightpos;
            features[index++] = adeprel_rightpos;
            int adeprel_leftsiblingw = (leftsiblingw<< 10) | adeprel;
            features[index++] = adeprel_leftsiblingw;
            int adeprel_leftsiblingpos = (adeprel<<10) | leftsiblingpos;
            features[index++] = adeprel_leftsiblingpos;
            int adeprel_rightsiblingw = (rightsiblingw<< 10) | adeprel;
            features[index++] = adeprel_rightsiblingw;
            int adeprel_rightsiblingpos = (adeprel<<10) | rightsiblingpos;
            features[index++] = adeprel_rightsiblingpos;


            String deprelpath_pospath = deprelpath+" "+ pospath;
            features[index++] = deprelpath_pospath;
            String deprelpath_position = deprelpath + " " +position;
            features[index++] = deprelpath_position;
            String deprelpath_leftw = leftw +" "+ deprelpath;
            features[index++] = deprelpath_leftw;
            String deprelpath_leftpos = deprelpath + " " +leftpos;
            features[index++] = deprelpath_leftpos;
            String deprelpath_rightw = rightw + " "+deprelpath;
            features[index++] = deprelpath_rightw;
            String deprelpath_rightpos = deprelpath +" " +rightpos;
            features[index++] = deprelpath_rightpos;
            String deprelpath_leftsiblingw = leftsiblingw+" "+deprelpath;
            features[index++] = deprelpath_leftsiblingw;
            String deprelpath_leftsiblingpos = deprelpath +" "+ leftsiblingpos;
            features[index++] = deprelpath_leftsiblingpos;
            String deprelpath_rightsiblingw = rightsiblingw+" "+deprelpath;
            features[index++] = deprelpath_rightsiblingw;
            String deprelpath_rightsiblingpos = deprelpath+" "+rightsiblingpos;
            features[index++] = deprelpath_rightsiblingpos;


            String pospath_position = pospath + " " +position;
            features[index++] = pospath_position;
            String pospath_leftw = leftw +" "+ pospath;
            features[index++] = pospath_leftw;
            String pospath_leftpos = pospath + " " +leftpos;
            features[index++] = pospath_leftpos;
            String pospath_rightw = rightw + " "+pospath;
            features[index++] = pospath_rightw;
            String pospath_rightpos = pospath +" " +rightpos;
            features[index++] = pospath_rightpos;
            String pospath_leftsiblingw = leftsiblingw+" "+pospath;
            features[index++] = pospath_leftsiblingw;
            String pospath_leftsiblingpos = pospath +" "+ leftsiblingpos;
            features[index++] = pospath_leftsiblingpos;
            String pospath_rightsiblingw = rightsiblingw+" "+pospath;
            features[index++] = pospath_rightsiblingw;
            String pospath_rightsiblingpos = pospath+" "+rightsiblingpos;
            features[index++] = pospath_rightsiblingpos;

            int position_leftw = (leftw<< 2) | position;
            features[index++] = position_leftw;
            int position_leftpos = (leftpos<<2) | position;
            features[index++] = position_leftpos;
            int position_rightw = (rightw<< 2) | position;
            features[index++] = position_rightw;
            int position_rightpos = (rightpos<<2) | position;
            features[index++] = position_rightpos;
            int position_leftsiblingw = (leftsiblingw<< 2) | position;
            features[index++] = position_leftsiblingw;
            int position_leftsiblingpos = (leftsiblingpos<<2) | position;
            features[index++] = position_leftsiblingpos;
            int position_rightsiblingw = (rightsiblingw<< 2) | position;
            features[index++] = position_rightsiblingw;
            int position_rightsiblingpos = (rightsiblingpos<<2) | position;
            features[index++] = position_rightsiblingpos;

            int leftw_leftpos = (leftw<<10) | leftpos;
            features[index++] = leftw_leftpos;
            long leftw_rightw = (leftw<< 20) | rightw;
            features[index++] = leftw_rightw;
            int leftw_rightpos = (leftw<<10) | rightpos;
            features[index++] = leftw_rightpos;
            long leftw_leftsiblingw = (leftw<< 20) | leftsiblingw;
            features[index++] = leftw_leftsiblingw;
            int leftw_leftsiblingpos = (leftw<<10) | leftsiblingpos;
            features[index++] = leftw_leftsiblingpos;
            long leftw_rightsiblingw = (leftw<< 20) | rightsiblingw;
            features[index++] = leftw_rightsiblingw;
            int leftw_rightsiblingpos = (leftw<<10) | rightsiblingpos;
            features[index++] = leftw_rightsiblingpos;

            int leftpos_rightw = (rightw<< 10) | leftpos;
            features[index++] = leftpos_rightw;
            int leftpos_rightpos = (leftpos<<10) | rightpos;
            features[index++] = leftpos_rightpos;
            int leftpos_leftsiblingw = (leftsiblingw<< 10) | leftpos;
            features[index++] = leftpos_leftsiblingw;
            int leftpos_leftsiblingpos = (leftpos<<10) | leftsiblingpos;
            features[index++] = leftpos_leftsiblingpos;
            int leftpos_rightsiblingw = (rightsiblingw<< 10) | leftpos;
            features[index++] = leftpos_rightsiblingw;
            int leftpos_rightsiblingpos = (leftpos<<10) | rightsiblingpos;
            features[index++] = leftpos_rightsiblingpos;

            int rightw_rightpos = (rightw<<10) | rightpos;
            features[index++] = rightw_rightpos;
            long rightw_leftsiblingw = (rightw<< 20) | leftsiblingw;
            features[index++] = rightw_leftsiblingw;
            int rightw_leftsiblingpos = (rightw<<10) | leftsiblingpos;
            features[index++] = rightw_leftsiblingpos;
            long rightw_rightsiblingw = (rightw<< 20) | rightsiblingw;
            features[index++] = rightw_rightsiblingw;
            int rightw_rightsiblingpos = (rightw<<10) | rightsiblingpos;
            features[index++] = rightw_rightsiblingpos;


            int rightpos_leftsiblingw = (leftsiblingw<< 10) | rightpos;
            features[index++] = rightpos_leftsiblingw;
            int rightpos_leftsiblingpos = (rightpos<<10) | leftsiblingpos;
            features[index++] = rightpos_leftsiblingpos;
            int rightpos_rightsiblingw = (rightsiblingw<< 10) | rightpos;
            features[index++] = rightpos_rightsiblingw;
            int rightpos_rightsiblingpos = (rightpos<<10) | rightsiblingpos;
            features[index++] = rightpos_rightsiblingpos;


            int leftsiblingw_leftsiblingpos = (leftsiblingw<<10) | leftsiblingpos;
            features[index++] = leftsiblingw_leftsiblingpos;
            long leftsiblingw_rightsiblingw = (leftsiblingw<< 20) | rightsiblingw;
            features[index++] = leftsiblingw_rightsiblingw;
            int leftsiblingw_rightsiblingpos = (leftsiblingw<<10) | rightsiblingpos;
            features[index++] = leftsiblingw_rightsiblingpos;

            long leftsiblingpos_rightsiblingw = (rightsiblingw<< 10) | leftsiblingpos;
            features[index++] = leftsiblingpos_rightsiblingw;
            int leftsiblingpos_rightsiblingpos = (rightsiblingpos<<10) | leftsiblingpos;
            features[index++] = leftsiblingpos_rightsiblingpos;

            int rightSiblingw_rightSiblingpos = (rightsiblingw<<10) | rightsiblingpos;
            features[index++] = rightSiblingw_rightSiblingpos;
            */
            //////////////////////////////////////////////////////////////////////
            ///////////////// PREDICATE-ARGUMENT CONJOINED FEATURES (154) ////////
            //////////////////////////////////////////////////////////////////////

            // todo 20 bits for words, 10 bits for pos, 10 bits for dependencies
            // todo e.g. (pw<<20) | aw ==> 20+20> 32 ==> long
            // pw + argument features
            long pw_aw = (pw<<20) | aw;
            features[index++] = pw_aw;
            // todo (pw<<10) | pos ==> 10+20<32 ==> int
            int pw_apos = (pw<<10) | apos ;
            features[index++] = pw_apos;
            int pw_adeprel = (pw<<10) | adeprel;
            features[index++] = pw_adeprel;
            //StringBuilder pw_deprelpath= new StringBuilder();
            //pw_deprelpath.append(pw);
            //pw_deprelpath.append(" ");
            //pw_deprelpath.append(deprelpath);
            String pw_deprelpath = pw +" "+deprelpath;
            features[index++] = pw_deprelpath;
            //StringBuilder pw_pospath= new StringBuilder();
            //pw_deprelpath.append(pw);
            //pw_deprelpath.append(" ");
            //pw_deprelpath.append(pospath);
            String pw_pospath = pw +" "+pospath;
            features[index++] = pw_pospath;
            int pw_position = (pw<<2) | position;
            features[index++] = pw_position;
            long pw_leftw = (pw<<20) | leftw;
            features[index++] = pw_leftw;
            int pw_leftpos = (pw<<10) | leftpos;
            features[index++] = pw_leftpos;
            long pw_rightw = (pw<<20) | rightw;
            features[index++] = pw_rightw;
            int pw_rightpos = (pw<<10) | rightpos;
            features[index++] = pw_rightpos;
            long pw_leftsiblingw = (pw<<20) | leftsiblingw;
            features[index++] = pw_leftsiblingw;
            int pw_leftsiblingpos= (pw<<10) | leftsiblingpos;
            features[index++] = pw_leftsiblingpos;
            long pw_rightsiblingw = (pw<<20) | rightsiblingw;
            features[index++] = pw_rightsiblingw;
            int pw_rightsiblingpos = (pw<<10) | rightsiblingpos;
            features[index++] = pw_rightsiblingpos;

            //ppos + argument features
            int aw_ppos = (aw<<10) | ppos;
            features[index++] = aw_ppos;
            int ppos_apos = (ppos<<10) | apos;
            features[index++] = ppos_apos;
            int ppos_adeprel = (ppos<<10) | adeprel;
            features[index++] = ppos_adeprel;
            //StringBuilder ppos_deprelpath = new StringBuilder();
            //ppos_deprelpath.append(ppos);
            //ppos_deprelpath.append(" ");
            //ppos_deprelpath.append(deprelpath);
            String ppos_deprelpath = ppos+" "+deprelpath;
            features[index++] = ppos_deprelpath;
            //StringBuilder ppos_pospath = new StringBuilder();
            //ppos_pospath.append(ppos);
            //ppos_pospath.append(" ");
            //ppos_pospath.append(pospath);
            String ppos_pospath= ppos +" "+pospath;
            features[index++] = ppos_pospath;
            int ppos_position = (ppos<<2) | position;
            features[index++] = ppos_position;
            int leftw_ppos = (leftw<<10) | ppos;
            features[index++] = leftw_ppos;
            int ppos_leftpos = (ppos<<10) | leftpos;
            features[index++] = ppos_leftpos;
            int rightw_ppos = (rightw<<10) | ppos;
            features[index++] = rightw_ppos;
            int ppos_rightpos = (ppos<<10) | rightpos;
            features[index++] = ppos_rightpos;
            int leftsiblingw_ppos = (leftsiblingw<<10) | ppos;
            features[index++] = leftsiblingw_ppos;
            int ppos_leftsiblingpos = (ppos<<10) | leftsiblingpos;
            features[index++] = ppos_leftsiblingpos;
            int rightsiblingw_ppos = (rightsiblingw<<10) | ppos;
            features[index++] = rightsiblingw_ppos;
            int ppos_rightsiblingpos = (ppos<<10) | rightsiblingpos;
            features[index++] = ppos_rightsiblingpos;


            //pdeprel + argument features
            int aw_pdeprel = (aw<<10) | pdeprel;
            features[index++] = aw_pdeprel;
            int pdeprel_apos = (pdeprel<<10) | apos;
            features[index++] = pdeprel_apos;
            int pdeprel_adeprel = (pdeprel<<10) | adeprel;
            features[index++] = pdeprel_adeprel;
            //StringBuilder pdeprel_deprelpath = new StringBuilder();
            //pdeprel_deprelpath.append(pdeprel);
            //pdeprel_deprelpath.append(" ");
            //pdeprel_deprelpath.append(deprelpath);
            String pdeprel_deprelpath = pdeprel + " "+deprelpath;
            features[index++] = pdeprel_deprelpath;
            //StringBuilder pdeprel_pospath= new StringBuilder();
            //pdeprel_pospath.append(pdeprel);
            //pdeprel_pospath.append(" ");
            //pdeprel_pospath.append(pospath);
            String pdeprel_pospath = pdeprel +" "+pospath;
            features[index++] = pdeprel_pospath;
            int pdeprel_position = (pdeprel<<2) | position;
            features[index++] = pdeprel_position;
            int leftw_pdeprel = (leftw<<10) | pdeprel;
            features[index++] = leftw_pdeprel;
            int pdeprel_leftpos = (pdeprel<<10) | leftpos;
            features[index++] = pdeprel_leftpos;
            int rightw_pdeprel = (rightw<<10) | pdeprel;
            features[index++] = rightw_pdeprel;
            int pdeprel_rightpos = (pdeprel<<10) | rightpos;
            features[index++] = pdeprel_rightpos;
            int leftsiblingw_pdeprel = (leftsiblingw<<10) |pdeprel;
            features[index++] = leftsiblingw_pdeprel;
            int pdeprel_leftsiblingpos = (pdeprel<<10) | leftsiblingpos;
            features[index++] = pdeprel_leftsiblingpos;
            int rightsiblingw_pdeprel = (rightsiblingw<<10) | pdeprel;
            features[index++] = rightsiblingw_pdeprel;
            int pdeprel_rightsiblingpos = (pdeprel<<10) | rightsiblingpos;
            features[index++] = pdeprel_rightsiblingpos;


            //plem + argument features
            long aw_plem = (aw<<20) | plem;
            features[index++] = aw_plem;
            int plem_apos = (plem<<10) | apos;
            features[index++] = plem_apos;
            int plem_adeprel = (plem<<10) | adeprel;
            features[index++] = plem_adeprel;
            //StringBuilder plem_deprelpath = new StringBuilder();
            //plem_deprelpath.append(plem);
            //plem_deprelpath.append(" ");
            //plem_deprelpath.append(deprelpath);
            String plem_deprelpath = plem +" "+deprelpath;
            features[index++] = plem_deprelpath;
            //StringBuilder plem_pospath= new StringBuilder();
            //plem_pospath.append(plem);
            //plem_pospath.append(" ");
            //plem_pospath.append(pospath);
            String plem_pospath = plem +" "+pospath;
            features[index++] = plem_pospath;
            int plem_position = (plem<<2) | position;
            features[index++] = plem_position;
            long leftw_plem = (leftw<<20) | plem;
            features[index++] = leftw_plem;
            int plem_leftpos = (plem<<10) | leftpos;
            features[index++] = plem_leftpos;
            long rightw_plem = (rightw<<20) | plem;
            features[index++] = rightw_plem;
            int plem_rightpos = (plem<<10) | rightpos;
            features[index++] = plem_rightpos;
            long leftsiblingw_plem = (leftsiblingw<<20) |plem;
            features[index++] = leftsiblingw_plem;
            int plem_leftsiblingpos = (plem<<10) | leftsiblingpos;
            features[index++] = plem_leftsiblingpos;
            long rightsiblingw_plem = (rightsiblingw<<20) | plem;
            features[index++] = rightsiblingw_plem;
            int plem_rightsiblingpos = (plem<<10) | rightsiblingpos;
            features[index++] = plem_rightsiblingpos;

            //psense + argument features
            //StringBuilder psense_aw = new StringBuilder();
            //psense_aw.append(psense);
            //psense_aw.append(" ");
            //psense_aw.append(aw);
            String psense_aw = pSense + " " + aw;
            features[index++] = psense_aw;
            //StringBuilder psense_apos = new StringBuilder();
            //psense_apos.append(psense);
            //psense_apos.append(" ");
            //psense_apos.append(apos);
            String psense_apos = pSense +" "+apos;
            features[index++] = psense_apos;
            //StringBuilder psense_adeprel = new StringBuilder();
            //psense_adeprel.append(psense);
            //psense_adeprel.append(" ");
            //psense_adeprel.append(adeprel);
            String psense_adeprel = pSense + " "+ adeprel;
            features[index++] = psense_adeprel;
            //StringBuilder psense_deprelpath = new StringBuilder();
            //psense_deprelpath.append(psense);
            //psense_deprelpath.append(" ");
            //psense_deprelpath.append(deprelpath);
            String psense_deprelpath = pSense + " "+ deprelpath;
            features[index++] = psense_deprelpath;
            //StringBuilder psense_pospath = new StringBuilder();
            //psense_pospath.append(psense);
            //psense_pospath.append(" ");
            //psense_pospath.append(pospath);
            String psense_pospath = pSense + " "+ pospath;
            features[index++] = psense_pospath;
            //StringBuilder psense_position = new StringBuilder();
            //psense_position.append(psense);
            //psense_position.append(" ");
            //psense_position.append(position);
            String psense_position = pSense + " " +position;
            features[index++] = psense_position;
            //StringBuilder psense_leftw = new StringBuilder();
            //psense_leftw.append(psense);
            //psense_leftw.append(" ");
            //psense_leftw.append(leftw);
            String psense_leftw = pSense + " " + leftw;
            features[index++] = psense_leftw;
            //StringBuilder psense_leftpos = new StringBuilder();
            //psense_leftpos.append(psense);
            //psense_leftpos.append(" ");
            //psense_leftpos.append(leftpos);
            String psense_leftpos = pSense + " " + leftpos;
            features[index++] = psense_leftpos;
            //StringBuilder psense_rightw = new StringBuilder();
            //psense_rightw.append(psense);
            //psense_rightw.append(" ");
            //psense_rightw.append(rightw);
            String psense_rightw = pSense + " " + rightw;
            features[index++] = psense_rightw;
            //StringBuilder psense_rightpos = new StringBuilder();
            //psense_rightpos.append(psense);
            //psense_rightpos.append(" ");
            //psense_rightpos.append(rightpos);
            String psense_rightpos = pSense + " " + rightpos;
            features[index++] = psense_rightpos;
            //StringBuilder psense_leftsiblingw = new StringBuilder();
            //psense_leftsiblingw.append(psense);
            //psense_leftsiblingw.append(" ");
            //psense_leftsiblingw.append(leftsiblingw);
            String psense_leftsiblingw = pSense +" " + leftsiblingw;
            features[index++] = psense_leftsiblingw;
            //StringBuilder psense_leftsiblingpos = new StringBuilder();
            //psense_leftsiblingpos.append(psense);
            //psense_leftsiblingpos.append(" ");
            //psense_leftsiblingpos.append(leftsiblingpos);
            String psense_leftsiblingpos = pSense + " " + leftsiblingpos;
            features[index++] = psense_leftsiblingpos;
            //StringBuilder psense_rightsiblingw = new StringBuilder();
            //psense_rightsiblingw.append(psense);
            //psense_rightsiblingw.append(" ");
            //psense_rightsiblingw.append(rightsiblingw);
            String psense_rightsiblingw = pSense +" "+rightsiblingw;
            features[index++] = psense_rightsiblingw;
            //StringBuilder psense_rightsiblingpos = new StringBuilder();
            //psense_rightsiblingpos.append(psense);
            //psense_rightsiblingpos.append(psense);
            //psense_rightsiblingpos.append(" ");
            //psense_rightsiblingpos.append(rightsiblingpos);
            String psense_rightsiblingpos = pSense +" "+rightsiblingpos;
            features[index++] = psense_rightsiblingpos;


            //pprw  + argument features
            long aw_pprw = (aw<<20) | pprw;
            features[index++] = aw_pprw;
            int pprw_apos = (pprw<<10) | apos;
            features[index++] = pprw_apos;
            int pprw_adeprel = (pprw<<10) | adeprel;
            features[index++] = pprw_adeprel;
            //StringBuilder pprw_deprelpath = new StringBuilder();
            //pprw_deprelpath.append(pprw);
            //pprw_deprelpath.append(" ");
            //pprw_deprelpath.append(deprelpath);
            String pprw_deprelpath = pprw +" "+deprelpath;
            features[index++] = pprw_deprelpath;
            //StringBuilder pprw_pospath= new StringBuilder();
            //pprw_pospath.append(pprw);
            //pprw_pospath.append(" ");
            //pprw_pospath.append(pospath);
            String pprw_pospath = pprw +" "+ pospath;
            features[index++] = pprw_pospath;
            int pprw_position = (pprw<<2) | position;
            features[index++] = pprw_position;
            long leftw_pprw = (leftw<<20) | pprw;
            features[index++] = leftw_pprw;
            int pprw_leftpos = (pprw<<10) | leftpos;
            features[index++] = pprw_leftpos;
            long rightw_pprw = (rightw<<20) | pprw;
            features[index++] = rightw_pprw;
            int pprw_rightpos = (pprw<<10) | rightpos;
            features[index++] = pprw_rightpos;
            long leftsiblingw_pprw = (leftsiblingw<<20) |pprw;
            features[index++] = leftsiblingw_pprw;
            int pprw_leftsiblingpos = (pprw<<10) | leftsiblingpos;
            features[index++] = pprw_leftsiblingpos;
            long rightsiblingw_pprw = (rightsiblingw<<20) | pprw;
            features[index++] = rightsiblingw_pprw;
            int pprw_rightsiblingpos = (pprw<<10) | rightsiblingpos;
            features[index++] = pprw_rightsiblingpos;


            //pdeprel + argument features
            int aw_pprpos = (aw<<10) | pprpos;
            features[index++] = aw_pprpos;
            int pprpos_apos = (pprpos<<10) | apos;
            features[index++] = pprpos_apos;
            int pprpos_adeprel = (pprpos<<10) | adeprel;
            features[index++] = pprpos_adeprel;
            //StringBuilder pprpos_deprelpath = new StringBuilder();
            //pprpos_deprelpath.append(pprpos);
            //pprpos_deprelpath.append(" ");
            //pprpos_deprelpath.append(deprelpath);
            String pprpos_deprelpath = pprpos +" "+deprelpath;
            features[index++] = pprpos_deprelpath;
            //StringBuilder pprpos_pospath= new StringBuilder();
            //pprpos_pospath.append(pprpos);
            //pprpos_pospath.append(" ");
            //pprpos_pospath.append(pospath);
            String pprpos_pospath= pprpos +" "+pospath;
            features[index++] = pprpos_pospath;
            int pprpos_position = (pprpos<<2) | position;
            features[index++] = pprpos_position;
            int leftw_pprpos = (leftw<<10) | pprpos;
            features[index++] = leftw_pprpos;
            int pprpos_leftpos = (pprpos<<10) | leftpos;
            features[index++] = pprpos_leftpos;
            int rightw_pprpos = (rightw<<10) | pprpos;
            features[index++] = rightw_pprpos;
            int pprpos_rightpos = (pprpos<<10) | rightpos;
            features[index++] = pprpos_rightpos;
            int leftsiblingw_pprpos = (leftsiblingw<<10) |pprpos;
            features[index++] = leftsiblingw_pprpos;
            int pprpos_leftsiblingpos = (pprpos<<10) | leftsiblingpos;
            features[index++] = pprpos_leftsiblingpos;
            int rightsiblingw_pprpos = (rightsiblingw<<10) | pprpos;
            features[index++] = rightsiblingw_pprpos;
            int pprpos_rightsiblingpos = (pprpos<<10) | rightsiblingpos;
            features[index++] = pprpos_rightsiblingpos;

            //pchilddepset + argument features
            //StringBuilder pchilddepset_aw = new StringBuilder();
            //pchilddepset_aw.append(pchilddepset);
            //pchilddepset_aw.append(" ");
            //pchilddepset_aw.append(aw);
            String pchilddepset_aw = pchilddepset +" "+aw;
            features[index++] = pchilddepset_aw;
            //StringBuilder pchilddepset_apos = new StringBuilder();
            //pchilddepset_apos.append(pchilddepset);
            //pchilddepset_apos.append(" ");
            //pchilddepset_apos.append(apos);
            String pchilddepset_apos = pchilddepset +" "+ apos;
            features[index++] = pchilddepset_apos;
            //StringBuilder pchilddepset_adeprel = new StringBuilder();
            //pchilddepset_adeprel.append(pchilddepset);
            //pchilddepset_adeprel.append(" ");
            //pchilddepset_adeprel.append(adeprel);
            String pchilddepset_adeprel = pchilddepset +" "+ adeprel;
            features[index++] = pchilddepset_adeprel;
            //StringBuilder pchilddepset_deprelpath = new StringBuilder();
            //pchilddepset_deprelpath.append(pchilddepset);
            //pchilddepset_deprelpath.append(" ");
            //pchilddepset_deprelpath.append(deprelpath);
            String pchilddepset_deprelpath = pchilddepset +" "+ deprelpath;
            features[index++] = pchilddepset_deprelpath;
            //StringBuilder pchilddepset_pospath = new StringBuilder();
            //pchilddepset_pospath.append(pchilddepset);
            //pchilddepset_pospath.append(" ");
            //pchilddepset_pospath.append(pospath);
            String pchilddepset_pospath = pchilddepset +" "+pospath;
            features[index++] = pchilddepset_pospath;
            //StringBuilder pchilddepset_position = new StringBuilder();
            //pchilddepset_position.append(pchilddepset);
            //pchilddepset_position.append(" ");
            //pchilddepset_position.append(position);
            String pchilddepset_position = pchilddepset +" "+position;
            features[index++] = pchilddepset_position;
            //StringBuilder pchilddepset_leftw = new StringBuilder();
            //pchilddepset_leftw.append(pchilddepset);
            //pchilddepset_leftw.append(" ");
            //pchilddepset_leftw.append(leftw);
            String pchilddepset_leftw = pchilddepset +" "+ leftw;
            features[index++] = pchilddepset_leftw;
            //StringBuilder pchilddepset_leftpos = new StringBuilder();
            //pchilddepset_leftpos.append(pchilddepset);
            //pchilddepset_leftpos.append(" ");
            //pchilddepset_leftpos.append(leftpos);
            String pchilddepset_leftpos = pchilddepset +" "+leftpos;
            features[index++] = pchilddepset_leftpos;
            //StringBuilder pchilddepset_rightw = new StringBuilder();
            //pchilddepset_rightw.append(pchilddepset);
            //pchilddepset_rightw.append(" ");
            //pchilddepset_rightw.append(rightw);
            String pchilddepset_rightw = pchilddepset +" "+rightw;
            features[index++] = pchilddepset_rightw;
            //StringBuilder pchilddepset_rightpos = new StringBuilder();
            //pchilddepset_rightpos.append(pchilddepset);
            //pchilddepset_rightpos.append(" ");
            //pchilddepset_rightpos.append(rightpos);
            String pchilddepset_rightpos = pchilddepset +" "+rightpos;
            features[index++] = pchilddepset_rightpos;
            //StringBuilder pchilddepset_leftsiblingw = new StringBuilder();
            //pchilddepset_leftsiblingw.append(pchilddepset);
            //pchilddepset_leftsiblingw.append(" ");
            //pchilddepset_leftsiblingw.append(leftsiblingw);
            String pchilddepset_leftsiblingw = pchilddepset +" "+leftsiblingw;
            features[index++] = pchilddepset_leftsiblingw;
            //StringBuilder pchilddepset_leftsiblingpos = new StringBuilder();
            //pchilddepset_leftsiblingpos.append(pchilddepset);
            //pchilddepset_leftsiblingpos.append(" ");
            //pchilddepset_leftsiblingpos.append(leftsiblingpos);
            String pchilddepset_leftsiblingpos = pchilddepset +" "+ leftsiblingpos;
            features[index++] = pchilddepset_leftsiblingpos;
            //StringBuilder pchilddepset_rightsiblingw = new StringBuilder();
            //pchilddepset_rightsiblingw.append(pchilddepset);
            //pchilddepset_rightsiblingw.append(" ");
            //pchilddepset_rightsiblingw.append(rightsiblingw);
            String pchilddepset_rightsiblingw = pchilddepset +" "+rightsiblingw;
            features[index++] = pchilddepset_rightsiblingw;
            //StringBuilder pchilddepset_rightsiblingpos = new StringBuilder();
            //pchilddepset_rightsiblingpos.append(pchilddepset);
            //pchilddepset_rightsiblingpos.append(" ");
            //pchilddepset_rightsiblingpos.append(rightsiblingpos);
            String pchilddepset_rightsiblingpos = pchilddepset + " "+rightsiblingpos;
            features[index++] = pchilddepset_rightsiblingpos;


            //pdepsubcat + argument features
            //StringBuilder pdepsubcat_aw = new StringBuilder();
            //pdepsubcat_aw.append(pdepsubcat);
            //pdepsubcat_aw.append(" ");
            //pdepsubcat_aw.append(aw);
            String pdepsubcat_aw =pdepsubcat +" " + aw;
            features[index++] = pdepsubcat_aw;
            //StringBuilder pdepsubcat_apos = new StringBuilder();
            //pdepsubcat_apos.append(pdepsubcat);
            //pdepsubcat_apos.append(" ");
            //pdepsubcat_apos.append(apos);
            String pdepsubcat_apos = pdepsubcat +" " + apos;
            features[index++] = pdepsubcat_apos;
            //StringBuilder pdepsubcat_adeprel = new StringBuilder();
            //pdepsubcat_adeprel.append(pdepsubcat);
            //pdepsubcat_adeprel.append(" ");
            //pdepsubcat_adeprel.append(adeprel);
            String pdepsubcat_adeprel = pdepsubcat +" "+adeprel;
            features[index++] = pdepsubcat_adeprel;
            //StringBuilder pdepsubcat_deprelpath = new StringBuilder();
            //pdepsubcat_deprelpath.append(pdepsubcat);
            //pdepsubcat_deprelpath.append(" ");
            //pdepsubcat_deprelpath.append(deprelpath);
            String pdepsubcat_deprelpath = pdepsubcat +" "+deprelpath;
            features[index++] = pdepsubcat_deprelpath;
            //StringBuilder pdepsubcat_pospath = new StringBuilder();
            //pdepsubcat_pospath.append(pdepsubcat);
            //pdepsubcat_pospath.append(" ");
            //pdepsubcat_pospath.append(pospath);
            String pdepsubcat_pospath = pdepsubcat +" "+pospath;
            features[index++] = pdepsubcat_pospath;
            //StringBuilder pdepsubcat_position = new StringBuilder();
            //pdepsubcat_position.append(pdepsubcat);
            //pdepsubcat_position.append(" ");
            //pdepsubcat_position.append(position);
            String pdepsubcat_position = pdepsubcat +" "+position;
            features[index++] = pdepsubcat_position;
            //StringBuilder pdepsubcat_leftw = new StringBuilder();
            //pdepsubcat_leftw.append(pdepsubcat);
            //pdepsubcat_leftw.append(" ");
            //pdepsubcat_leftw.append(leftw);
            String pdepsubcat_leftw = pdepsubcat +" "+leftw;
            features[index++] = pdepsubcat_leftw;
            //StringBuilder pdepsubcat_leftpos = new StringBuilder();
            //pdepsubcat_leftpos.append(pdepsubcat);
            //pdepsubcat_leftpos.append(" ");
            //pdepsubcat_leftpos.append(leftpos);
            String pdepsubcat_leftpos =pdepsubcat +" "+ leftpos;
            features[index++] = pdepsubcat_leftpos;
            //StringBuilder pdepsubcat_rightw = new StringBuilder();
            //pdepsubcat_rightw.append(pdepsubcat);
            //pdepsubcat_rightw.append(" ");
            //pdepsubcat_rightw.append(rightw);
            String pdepsubcat_rightw = pdepsubcat +" "+ rightw;
            features[index++] = pdepsubcat_rightw;
            //StringBuilder pdepsubcat_rightpos = new StringBuilder();
            //pdepsubcat_rightpos.append(pdepsubcat);
            //pdepsubcat_rightpos.append(" ");
            //pdepsubcat_rightpos.append(rightpos);
            String pdepsubcat_rightpos = pdepsubcat +" "+rightpos;
            features[index++] = pdepsubcat_rightpos;
            //StringBuilder pdepsubcat_leftsiblingw = new StringBuilder();
            //pdepsubcat_leftsiblingw.append(pdepsubcat);
            //pdepsubcat_leftsiblingw.append(" ");
            //pdepsubcat_leftsiblingw.append(leftsiblingw);
            String pdepsubcat_leftsiblingw =pdepsubcat +" "+ leftsiblingw;
            features[index++] = pdepsubcat_leftsiblingw;
            //StringBuilder pdepsubcat_leftsiblingpos = new StringBuilder();
            //pdepsubcat_leftsiblingpos.append(pdepsubcat);
            //pdepsubcat_leftsiblingpos.append(" ");
            //pdepsubcat_leftsiblingpos.append(leftsiblingpos);
            String pdepsubcat_leftsiblingpos = pdepsubcat + " "+leftsiblingpos;
            features[index++] = pdepsubcat_leftsiblingpos;
            //StringBuilder pdepsubcat_rightsiblingw = new StringBuilder();
            //pdepsubcat_rightsiblingw.append(pdepsubcat);
            //pdepsubcat_rightsiblingw.append(" ");
            //pdepsubcat_rightsiblingw.append(rightsiblingw);
            String pdepsubcat_rightsiblingw = pdepsubcat +" "+rightsiblingw;
            features[index++] = pdepsubcat_rightsiblingw;
            //StringBuilder pdepsubcat_rightsiblingpos = new StringBuilder();
            //pdepsubcat_rightsiblingpos.append(pdepsubcat);
            //pdepsubcat_rightsiblingpos.append(" ");
            //pdepsubcat_rightsiblingpos.append(rightsiblingpos);
            String pdepsubcat_rightsiblingpos = pdepsubcat +" "+rightsiblingpos;
            features[index++] = pdepsubcat_rightsiblingpos;


            //pchildposset + argument features
            //StringBuilder pchildposset_aw = new StringBuilder();
            ///pchildposset_aw.append(pchildposset);
            //pchildposset_aw.append(" ");
            //pchildposset_aw.append(aw);
            String pchildposset_aw = pchildposset + " "+ aw;
            features[index++] = pchildposset_aw;
            //StringBuilder pchildposset_apos = new StringBuilder();
            //pchildposset_apos.append(pchildposset);
            //pchildposset_apos.append(" ");
            //pchildposset_apos.append(apos);
            String pchildposset_apos = pchildposset +" "+apos;
            features[index++] = pchildposset_apos;
            //StringBuilder pchildposset_adeprel = new StringBuilder();
            //pchildposset_adeprel.append(pchildposset);
            //pchildposset_adeprel.append(" ");
            //pchildposset_adeprel.append(adeprel);
            String pchildposset_adeprel = pchildposset + " " + adeprel;
            features[index++] = pchildposset_adeprel;
            //StringBuilder pchildposset_deprelpath = new StringBuilder();
            //pchildposset_deprelpath.append(pchildposset);
            //pchildposset_deprelpath.append(" ");
            //pchildposset_deprelpath.append(deprelpath);
            String pchildposset_deprelpath =pchildposset +" "+deprelpath;
            features[index++] = pchildposset_deprelpath;
            //StringBuilder pchildposset_pospath = new StringBuilder();
            //pchildposset_pospath.append(pchildposset);
            //pchildposset_pospath.append(" ");
            //pchildposset_pospath.append(pospath);
            String pchildposset_pospath = pchildposset +" "+ pospath;
            features[index++] = pchildposset_pospath;
            //StringBuilder pchildposset_position = new StringBuilder();
            //pchildposset_position.append(pchildposset);
            //pchildposset_position.append(" ");
            //pchildposset_position.append(position);
            String pchildposset_position = pchildposset + " "+ position;
            features[index++] = pchildposset_position;
            //StringBuilder pchildposset_leftw = new StringBuilder();
            //pchildposset_leftw.append(pchildposset);
            //pchildposset_leftw.append(" ");
            //pchildposset_leftw.append(leftw);
            String pchildposset_leftw = pchildposset +" "+leftw;
            features[index++] = pchildposset_leftw;
            //StringBuilder pchildposset_leftpos = new StringBuilder();
            //pchildposset_leftpos.append(pchildposset);
            //pchildposset_leftpos.append(" ");
            //pchildposset_leftpos.append(leftpos);
            String pchildposset_leftpos = pchildposset +" "+ leftpos;
            features[index++] = pchildposset_leftpos;
            //StringBuilder pchildposset_rightw = new StringBuilder();
            //pchildposset_rightw.append(pchildposset);
            //pchildposset_rightw.append(" ");
            //pchildposset_rightw.append(rightw);
            String pchildposset_rightw = pchildposset + " "+rightw;
            features[index++] = pchildposset_rightw;
            //StringBuilder pchildposset_rightpos = new StringBuilder();
            //pchildposset_rightpos.append(pchildposset);
            //pchildposset_rightpos.append(" ");
            //pchildposset_rightpos.append(rightpos);
            String pchildposset_rightpos = pchildposset + " "+rightpos;
            features[index++] = pchildposset_rightpos;
            //StringBuilder pchildposset_leftsiblingw = new StringBuilder();
            //pchildposset_leftsiblingw.append(pchildposset);
            //pchildposset_leftsiblingw.append(" ");
            //pchildposset_leftsiblingw.append(leftsiblingw);
            String pchildposset_leftsiblingw = pchildposset + " "+ leftsiblingw;
            features[index++] = pchildposset_leftsiblingw;
            //StringBuilder pchildposset_leftsiblingpos = new StringBuilder();
            //pchildposset_leftsiblingpos.append(pchildposset);
            //pchildposset_leftsiblingpos.append(" ");
            //pchildposset_leftsiblingpos.append(leftsiblingpos);
            String pchildposset_leftsiblingpos = pchildposset + " "+ leftsiblingpos;
            features[index++] = pchildposset_leftsiblingpos;
            //StringBuilder pchildposset_rightsiblingw = new StringBuilder();
            //pchildposset_rightsiblingw.append(pchildposset);
            //pchildposset_rightsiblingw.append(" ");
            //pchildposset_rightsiblingw.append(rightsiblingw);
            String pchildposset_rightsiblingw = pchildposset +" " +rightsiblingw;
            features[index++] = pchildposset_rightsiblingw;
            //StringBuilder pchildposset_rightsiblingpos = new StringBuilder();
            //pchildposset_rightsiblingpos.append(pchildposset);
            //pchildposset_rightsiblingpos.append(" ");
            //pchildposset_rightsiblingpos.append(rightsiblingpos);
            String pchildposset_rightsiblingpos = pchildposset +" "+rightsiblingpos;
            features[index++] = pchildposset_rightsiblingpos;


            //pchildwset + argument features
            //StringBuilder pchildwset_aw = new StringBuilder();
            //pchildwset_aw.append(pchildwset);
            //pchildwset_aw.append(" ");
            //pchildwset_aw.append(aw);
            String pchildwset_aw = pchildwset +" "+ aw;
            features[index++] = pchildwset_aw;
            //StringBuilder pchildwset_apos = new StringBuilder();
            //pchildwset_apos.append(pchildwset);
            //pchildwset_apos.append(" ");
            //pchildwset_apos.append(apos);
            String pchildwset_apos = pchildwset + " "+apos;
            features[index++] = pchildwset_apos;
            //StringBuilder pchildwset_adeprel = new StringBuilder();
            //pchildwset_adeprel.append(pchildwset);
            //pchildwset_adeprel.append(" ");
            //pchildwset_adeprel.append(adeprel);
            String pchildwset_adeprel = pchildwset +" "+adeprel;
            features[index++] = pchildwset_adeprel;
            //StringBuilder pchildwset_deprelpath = new StringBuilder();
            //pchildwset_deprelpath.append(pchildwset);
            //pchildwset_deprelpath.append(" ");
            //pchildwset_deprelpath.append(deprelpath);
            String pchildwset_deprelpath = pchildwset +" "+deprelpath;
            features[index++] = pchildwset_deprelpath;
            //StringBuilder pchildwset_pospath = new StringBuilder();
            //pchildwset_pospath.append(pchildwset);
            //pchildwset_pospath.append(" ");
            //pchildwset_pospath.append(pospath);
            String pchildwset_pospath = pchildwset +" "+pospath;
            features[index++] = pchildwset_pospath;
            //StringBuilder pchildwset_position = new StringBuilder();
            //pchildwset_position.append(pchildwset);
            //pchildwset_position.append(" ");
            //pchildwset_position.append(position);
            String pchildwset_position = pchildwset +" "+ position;
            features[index++] = pchildwset_position;
            //StringBuilder pchildwset_leftw = new StringBuilder();
            //pchildwset_leftw.append(pchildwset);
            //pchildwset_leftw.append(" ");
            //pchildwset_leftw.append(leftw);
            String pchildwset_leftw = pchildwset +" "+leftw;
            features[index++] = pchildwset_leftw;
            //StringBuilder pchildwset_leftpos = new StringBuilder();
            //pchildwset_leftpos.append(pchildwset);
            //pchildwset_leftpos.append(" ");
            //pchildwset_leftpos.append(leftpos);
            String pchildwset_leftpos = pchildwset +" "+leftpos;
            features[index++] = pchildwset_leftpos;
            //StringBuilder pchildwset_rightw = new StringBuilder();
            //pchildwset_rightw.append(pchildwset);
            //pchildwset_rightw.append(" ");
            //pchildwset_rightw.append(rightw);
            String pchildwset_rightw = pchildwset +" "+rightw;
            features[index++] = pchildwset_rightw;
            //StringBuilder pchildwset_rightpos = new StringBuilder();
            //pchildwset_rightpos.append(pchildwset);
            //pchildwset_rightpos.append(" ");
            //pchildwset_rightpos.append(rightpos);
            String pchildwset_rightpos = pchildwset + " "+ rightpos;
            features[index++] = pchildwset_rightpos;
            //StringBuilder pchildwset_leftsiblingw = new StringBuilder();
            //pchildwset_leftsiblingw.append(pchildwset);
            //pchildwset_leftsiblingw.append(" ");
            //pchildwset_leftsiblingw.append(leftsiblingw);
            String pchildwset_leftsiblingw = pchildwset +" "+leftsiblingw;
            features[index++] = pchildwset_leftsiblingw;
            //StringBuilder pchildwset_leftsiblingpos = new StringBuilder();
            //pchildwset_leftsiblingpos.append(pchildwset);
            //pchildwset_leftsiblingpos.append(" ");
            //pchildwset_leftsiblingpos.append(leftsiblingpos);
            String pchildwset_leftsiblingpos = pchildwset +" "+leftsiblingpos;
            features[index++] = pchildwset_leftsiblingpos;
            //StringBuilder pchildwset_rightsiblingw = new StringBuilder();
            //pchildwset_rightsiblingw.append(pchildwset);
            //pchildwset_rightsiblingw.append(" ");
            //pchildwset_rightsiblingw.append(rightsiblingw);
            String pchildwset_rightsiblingw = pchildwset +" " + rightsiblingw;
            features[index++] = pchildwset_rightsiblingw;
            //StringBuilder pchildwset_rightsiblingpos = new StringBuilder();
            //pchildwset_rightsiblingpos.append(pchildwset);
            //pchildwset_rightsiblingpos.append(" ");
            //pchildwset_rightsiblingpos.append(rightsiblingpos);
            String pchildwset_rightsiblingpos = pchildwset +" "+rightsiblingpos;
            features[index++] = pchildwset_rightsiblingpos;

            //////////////////////////////////////////////////////////////////////
            /////////// PREDICATE-ARGUMENT-ARGUMENT CONJOINED FEATURES (91) //////
            //////////////////////////////////////////////////////////////////////

            ///////////////////////////
            /// Plem + arg-arg ///////
            //////////////////////////
            /*
            long plem_aw_apos = ((plem<<20) | aw)<<10 | apos;
            features[index++] = plem_aw_apos;
            long plem_aw_adeprel = ((plem<<20) | aw) <<10 | adeprel;
            features[index++] = plem_aw_adeprel;
            String plem_aw_deprelpath = plem+" "+aw+" "+ deprelpath;
            features[index++] = plem_aw_deprelpath;
            String plem_aw_pospath = plem+" "+aw+" "+ pospath;
            features[index++] = plem_aw_pospath;
            long plem_aw_position = ((plem<<20) | aw) <<2 | position;
            features[index++] = plem_aw_position;
            long plem_aw_leftw = ((plem<<20)| aw) << 20 | leftw;
            features[index++] = plem_aw_leftw;
            long plem_aw_leftpos = ((plem<<20) | aw )<<10 | leftpos;
            features[index++] = plem_aw_leftpos;
            long plem_aw_rightw = ((plem<<20) | aw ) << 20 | rightw;
            features[index++] = plem_aw_rightw;
            long plem_aw_rightpos = ((plem<<20) | aw) <<10 | rightpos;
            features[index++] = plem_aw_rightpos;
            long plem_aw_leftsiblingw = ((plem<<20) | aw )<< 20 | leftsiblingw;
            features[index++] = plem_aw_leftsiblingw;
            long plem_aw_leftsiblingpos = ((plem <<20) | aw) <<10 | leftsiblingpos;
            features[index++] = plem_aw_leftsiblingpos;
            long plem_aw_rightsiblingw = ((plem <<20 ) | aw) << 20 | rightsiblingw;
            features[index++] = plem_aw_rightsiblingw;
            long plem_aw_rightsiblingpos = ((plem<<20) |aw ) <<10 | rightsiblingpos;
            features[index++] = plem_aw_rightsiblingpos;

            long plem_apos_adeprel = ((plem <<10 ) | apos) <<10 | adeprel;
            features[index++] = plem_apos_adeprel;
            String plem_apos_deprelpath = plem+" "+apos+" "+ deprelpath;
            features[index++] = plem_apos_deprelpath;
            String plem_apos_pospath = plem+" "+apos+" "+ pospath;
            features[index++] = plem_apos_pospath;
            long plem_apos_position = ((plem<<10) | apos)<<2 | position;
            features[index++] = plem_apos_position;
            long plem_apos_leftw = ((plem <<20) | leftw)<< 10 | apos;
            features[index++] = plem_apos_leftw;
            long plem_apos_leftpos = ((plem <<10 ) | apos ) <<10 | leftpos;
            features[index++] = plem_apos_leftpos;
            long plem_apos_rightw = ((plem<<20 ) | rightw)<< 10 | apos;
            features[index++] = plem_apos_rightw;
            long plem_apos_rightpos = ((plem <<10 ) | apos )<<10 | rightpos;
            features[index++] = plem_apos_rightpos;
            long plem_apos_leftsiblingw = ((plem<<20) | leftsiblingw ) << 10 | apos;
            features[index++] = plem_apos_leftsiblingw;
            long plem_apos_leftsiblingpos = ((plem <<10 ) | apos )<<10 | leftsiblingpos;
            features[index++] = plem_apos_leftsiblingpos;
            long plem_apos_rightsiblingw = ((plem <<20 ) | rightsiblingw)<< 10 | apos;
            features[index++] = plem_apos_rightsiblingw;
            long plem_apos_rightsiblingpos = ((plem <<10 ) | apos ) <<10 | rightsiblingpos;
            features[index++] = plem_apos_rightsiblingpos;

            String plem_adeprel_deprelpath = plem+ " "+adeprel+" "+ deprelpath;
            features[index++] = plem_adeprel_deprelpath;
            String plem_adeprel_pospath = plem+" "+adeprel+" "+ pospath;
            features[index++] = plem_adeprel_pospath;
            long plem_adeprel_position = ((plem <<10 )| adeprel )<<2 | position;
            features[index++] = plem_adeprel_position;
            long plem_adeprel_leftw = ((plem <<20 ) | leftw ) << 10 | adeprel;
            features[index++] = plem_adeprel_leftw;
            long plem_adeprel_leftpos = ((plem<<10 ) | adeprel ) <<10 | leftpos;
            features[index++] = plem_adeprel_leftpos;
            long plem_adeprel_rightw = ((plem <<20) | rightw ) << 10 | adeprel;
            features[index++] = plem_adeprel_rightw;
            long plem_adeprel_rightpos = ((plem <<10 )| adeprel ) <<10 | rightpos;
            features[index++] = plem_adeprel_rightpos;
            long plem_adeprel_leftsiblingw = ((plem<<20 )| leftsiblingw ) << 10 | adeprel;
            features[index++] = plem_adeprel_leftsiblingw;
            long plem_adeprel_leftsiblingpos = ((plem << 10 ) | adeprel ) <<10 | leftsiblingpos;
            features[index++] = plem_adeprel_leftsiblingpos;
            long plem_adeprel_rightsiblingw = ((plem <<20 ) |rightsiblingw ) << 10 | adeprel;
            features[index++] = plem_adeprel_rightsiblingw;
            long plem_adeprel_rightsiblingpos = ((plem <<10 ) | adeprel) <<10 | rightsiblingpos;
            features[index++] = plem_adeprel_rightsiblingpos;


            String plem_deprelpath_pospath = plem +" "+deprelpath+" "+ pospath;
            features[index++] = plem_deprelpath_pospath;
            String plem_deprelpath_position = plem +" "+deprelpath + " " +position;
            features[index++] = plem_deprelpath_position;
            String plem_deprelpath_leftw = plem +" "+leftw +" "+ deprelpath;
            features[index++] = plem_deprelpath_leftw;
            String plem_deprelpath_leftpos = plem +" "+deprelpath + " " +leftpos;
            features[index++] = plem_deprelpath_leftpos;
            String plem_deprelpath_rightw = plem +" "+rightw + " "+deprelpath;
            features[index++] = plem_deprelpath_rightw;
            String plem_deprelpath_rightpos = plem +" "+deprelpath +" " +rightpos;
            features[index++] = plem_deprelpath_rightpos;
            String plem_deprelpath_leftsiblingw = plem +" "+leftsiblingw+" "+deprelpath;
            features[index++] = plem_deprelpath_leftsiblingw;
            String plem_deprelpath_leftsiblingpos = plem +" "+deprelpath +" "+ leftsiblingpos;
            features[index++] = plem_deprelpath_leftsiblingpos;
            String plem_deprelpath_rightsiblingw = plem +" "+rightsiblingw+" "+deprelpath;
            features[index++] = plem_deprelpath_rightsiblingw;
            String plem_deprelpath_rightsiblingpos = plem +" "+deprelpath+" "+rightsiblingpos;
            features[index++] = plem_deprelpath_rightsiblingpos;


            String plem_pospath_position = plem +" "+pospath + " " +position;
            features[index++] = plem_pospath_position;
            String plem_pospath_leftw = plem +" "+leftw +" "+ pospath;
            features[index++] = plem_pospath_leftw;
            String plem_pospath_leftpos = plem +" "+pospath + " " +leftpos;
            features[index++] = plem_pospath_leftpos;
            String plem_pospath_rightw = plem +" "+rightw + " "+pospath;
            features[index++] = plem_pospath_rightw;
            String plem_pospath_rightpos = plem +" "+pospath +" " +rightpos;
            features[index++] = plem_pospath_rightpos;
            String plem_pospath_leftsiblingw = plem +" "+leftsiblingw+" "+pospath;
            features[index++] = plem_pospath_leftsiblingw;
            String plem_pospath_leftsiblingpos =plem +" "+ pospath +" "+ leftsiblingpos;
            features[index++] = plem_pospath_leftsiblingpos;
            String plem_pospath_rightsiblingw = plem +" "+rightsiblingw+" "+pospath;
            features[index++] = plem_pospath_rightsiblingw;
            String plem_pospath_rightsiblingpos = plem +" "+pospath+" "+rightsiblingpos;
            features[index++] = plem_pospath_rightsiblingpos;

            long plem_position_leftw = ((plem<<20 ) | leftw )<< 2 | position;
            features[index++] = plem_position_leftw;
            long plem_position_leftpos = ((plem <<10 ) | leftpos)<<2 | position;
            features[index++] = plem_position_leftpos;
            long plem_position_rightw = ((plem <<20 ) | rightw ) << 2 | position;
            features[index++] = plem_position_rightw;
            long plem_position_rightpos = ((plem <<10 ) | rightpos )<<2 | position;
            features[index++] = plem_position_rightpos;
            long plem_position_leftsiblingw = ((plem<<20 ) | leftsiblingw ) << 2 | position;
            features[index++] = plem_position_leftsiblingw;
            long plem_position_leftsiblingpos = ((plem<<10 )| leftsiblingpos ) <<2 | position;
            features[index++] = plem_position_leftsiblingpos;
            long plem_position_rightsiblingw = ((plem<<20) | rightsiblingw ) << 2 | position;
            features[index++] = plem_position_rightsiblingw;
            long plem_position_rightsiblingpos = ((plem << 10 ) | rightsiblingpos ) <<2 | position;
            features[index++] = plem_position_rightsiblingpos;

            long plem_leftw_leftpos = ((plem<<20) | leftw ) <<10 | leftpos;
            features[index++] = plem_leftw_leftpos;
            long plem_leftw_rightw = ((plem <<20) | leftw ) << 20 | rightw;
            features[index++] = plem_leftw_rightw;
            long plem_leftw_rightpos = ((plem <<20) | leftw ) <<10 | rightpos;
            features[index++] = plem_leftw_rightpos;
            long plem_leftw_leftsiblingw = ((plem <<20 ) | leftw ) << 20 | leftsiblingw;
            features[index++] = plem_leftw_leftsiblingw;
            long plem_leftw_leftsiblingpos = ((plem<< 20 ) | leftw ) <<10 | leftsiblingpos;
            features[index++] = plem_leftw_leftsiblingpos;
            long plem_leftw_rightsiblingw = ((plem <<20 ) | leftw ) << 20 | rightsiblingw;
            features[index++] = plem_leftw_rightsiblingw;
            long plem_leftw_rightsiblingpos = ((plem <<20 ) | leftw )<<10 | rightsiblingpos;
            features[index++] = plem_leftw_rightsiblingpos;

            long plem_leftpos_rightw = ((plem <<20) | rightw ) << 10 | leftpos;
            features[index++] = plem_leftpos_rightw;
            long plem_leftpos_rightpos = ((plem << 10 ) | leftpos ) <<10 | rightpos;
            features[index++] = plem_leftpos_rightpos;
            long plem_leftpos_leftsiblingw = ((plem << 20) | leftsiblingw ) << 10 | leftpos;
            features[index++] = plem_leftpos_leftsiblingw;
            long plem_leftpos_leftsiblingpos = ((plem << 10) | leftpos ) <<10 | leftsiblingpos;
            features[index++] = plem_leftpos_leftsiblingpos;
            long plem_leftpos_rightsiblingw = ((plem << 20)| rightsiblingw ) << 10 | leftpos;
            features[index++] = plem_leftpos_rightsiblingw;
            long plem_leftpos_rightsiblingpos = ((plem << 10) | leftpos )<<10 | rightsiblingpos;
            features[index++] = plem_leftpos_rightsiblingpos;

            long plem_rightw_rightpos = ((plem <<20) | rightw ) <<10 | rightpos;
            features[index++] = plem_rightw_rightpos;
            long plem_rightw_leftsiblingw = ((plem <<20) | rightw ) << 20 | leftsiblingw;
            features[index++] = plem_rightw_leftsiblingw;
            long plem_rightw_leftsiblingpos = ((plem<< 20)| rightw ) <<10 | leftsiblingpos;
            features[index++] = plem_rightw_leftsiblingpos;
            long plem_rightw_rightsiblingw = (( plem<<20 ) | rightw ) << 20 | rightsiblingw;
            features[index++] = plem_rightw_rightsiblingw;
            long plem_rightw_rightsiblingpos = ((plem << 20 ) | rightw ) <<10 | rightsiblingpos;
            features[index++] = plem_rightw_rightsiblingpos;


            long plem_rightpos_leftsiblingw = (( plem << 20 ) | leftsiblingw ) << 10 | rightpos;
            features[index++] = plem_rightpos_leftsiblingw;
            long plem_rightpos_leftsiblingpos = (( plem << 10 ) | rightpos ) <<10 | leftsiblingpos;
            features[index++] = plem_rightpos_leftsiblingpos;
            long plem_rightpos_rightsiblingw = (( plem <<20 ) | rightsiblingw ) << 10 | rightpos;
            features[index++] = plem_rightpos_rightsiblingw;
            long plem_rightpos_rightsiblingpos = (( plem << 10) | rightpos) <<10 | rightsiblingpos;
            features[index++] = plem_rightpos_rightsiblingpos;


            long plem_leftsiblingw_leftsiblingpos = (( plem << 20) | leftsiblingw ) <<10 | leftsiblingpos;
            features[index++] = plem_leftsiblingw_leftsiblingpos;
            long plem_leftsiblingw_rightsiblingw = (( plem << 20) | leftsiblingw ) << 20 | rightsiblingw;
            features[index++] = plem_leftsiblingw_rightsiblingw;
            long plem_leftsiblingw_rightsiblingpos = ((plem << 20) | leftsiblingw ) <<10 | rightsiblingpos;
            features[index++] = plem_leftsiblingw_rightsiblingpos;

            long plem_leftsiblingpos_rightsiblingw = ((plem << 20) | rightsiblingw ) << 10 | leftsiblingpos;
            features[index++] = plem_leftsiblingpos_rightsiblingw;
            long plem_leftsiblingpos_rightsiblingpos = ((plem << 10) | rightsiblingpos ) <<10 | leftsiblingpos;
            features[index++] = plem_leftsiblingpos_rightsiblingpos;

            long plem_rightSiblingw_rightSiblingpos = (( plem << 20) | rightsiblingw ) <<10 | rightsiblingpos;
            features[index++] = plem_rightSiblingw_rightSiblingpos;

            //some miscellaneous tri-gram features

            int ppos_apos_adeprel = (((ppos<<10) | apos )<< 10) | adeprel;
            features[index++] = ppos_apos_adeprel;
            int pdeprel_apos_adeprel = (((pdeprel<<10) | apos )<< 10) | adeprel;
            features[index++] = pdeprel_apos_adeprel;

            String pchilddepset_apos_adeprel = pchilddepset +" "+apos+" "+adeprel;
            features[index++] = pchilddepset_apos_adeprel;

            String pchildposset_apos_adeprel = pchildposset +" "+apos +" "+adeprel;
            features[index++] = pchildposset_apos_adeprel;

            String pchildwset_apos_adeprel = pchildwset +" "+apos +" "+adeprel;
            features[index++] = pchildwset_apos_adeprel;

            String pchildwset_aw_adeprel = pchildwset +" "+aw +" "+ adeprel;
            features[index++] = pchildwset_aw_adeprel;
            */
        }
        else if (state.equals("joint")) {
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
            ///////////////// PREDICATE-PREDICATE CONJOINED FEATURES ////////////
            //////////////////////////////////////////////////////////////////////

            //todo 20 bits for words, 10 bits for pos, 10 bits for dependencies
            // pw, ppos, plem, pdeprel, pSense, pprw, pprpos, pdepsubcat, pchilddepset, pchildposset, pchildwset;
            /*
            //pw + ... (20 bits for pw)
            int pw_ppos = (pw<<10) | ppos ;
            features[index++] = pw_ppos;
            long pw_plem = (pw<<20) | plem;
            features[index++] = pw_plem;
            int pw_pdeprel = (pw<<10) | pdeprel;
            features[index++] = pw_pdeprel;
            String pw_psense = pw + " " + pSense;
            features[index++] = pw_psense;
            long pw_pprw = (pw<<20) | pprw;
            features[index++] = pw_pprw;
            int pw_pprpos = (pw<<10) | pprpos ;
            features[index++] = pw_pprpos;
            String pw_pdepsubcat = pw + " " + pdepsubcat;
            features[index++] = pw_pdepsubcat;
            String pw_pchilddepset = pw + " " + pchilddepset;
            features[index++] = pw_pchilddepset;
            String pw_pchildposset = pw + " " + pchildposset;
            features[index++] = pw_pchildposset;
            String pw_pchildwset = pw + " " + pchildwset;
            features[index++] = pw_pchildwset;

            //ppos + ...(10 bits for ppos)
            int ppos_plem = (plem<<10) | ppos;
            features[index++] = ppos_plem;
            int ppos_pdeprel = (ppos<<10) | pdeprel;
            features[index++] = ppos_pdeprel;
            String ppos_psense = ppos + " " + pSense;
            features[index++] = ppos_psense;
            int ppos_pprw = (pprw<<10) | ppos;
            features[index++] = ppos_pprw;
            int ppos_pprpos = (ppos<<10) | pprpos ;
            features[index++] = ppos_pprpos;
            String ppos_pdepsubcat = ppos + " " + pdepsubcat;
            features[index++] = ppos_pdepsubcat;
            String ppos_pchilddepset = ppos + " " + pchilddepset;
            features[index++] = ppos_pchilddepset;
            String ppos_pchildposset = ppos + " " + pchildposset;
            features[index++] = ppos_pchildposset;
            String ppos_pchildwset = ppos + " " + pchildwset;
            features[index++] = ppos_pchildwset;

            //plem + ... (20 bits for plem)
            int plem_pdeprel = (plem<<10) | pdeprel;
            features[index++] = plem_pdeprel;
            String plem_psense = plem + " " + pSense;
            features[index++] = plem_psense;
            long plem_pprw = (plem<<20) | pprw;
            features[index++] = plem_pprw;
            int plem_pprpos = (plem<<10) | pprpos ;
            features[index++] = plem_pprpos;
            String plem_pdepsubcat = plem + " " + pdepsubcat;
            features[index++] = plem_pdepsubcat;
            String plem_pchilddepset = plem + " " + pchilddepset;
            features[index++] = plem_pchilddepset;
            String plem_pchildposset = plem + " " + pchildposset;
            features[index++] = plem_pchildposset;
            String plem_pchildwset = plem + " " + pchildwset;
            features[index++] = plem_pchildwset;

            //pdeprel + ...(10 bits for pdeprel)
            String pdeprel_psense = pdeprel + " " + pSense;
            features[index++] = pdeprel_psense;
            int pdeprel_pprw = (pprw<<10) | pdeprel;
            features[index++] = pdeprel_pprw;
            int pdeprel_pprpos = (pdeprel<<10) | pprpos ;
            features[index++] = pdeprel_pprpos;
            String pdeprel_pdepsubcat = pdeprel + " " + pdepsubcat;
            features[index++] = pdeprel_pdepsubcat;
            String pdeprel_pchilddepset = pdeprel + " " + pchilddepset;
            features[index++] = pdeprel_pchilddepset;
            String pdeprel_pchildposset = pdeprel + " " + pchildposset;
            features[index++] = pdeprel_pchildposset;
            String pdeprel_pchildwset = pdeprel + " " + pchildwset;
            features[index++] = pdeprel_pchildwset;

            //psense + ...
            String psense_pprw = pSense+ " "+ pprw;
            features[index++] = psense_pprw;
            String psense_pprpos = pSense+" "+ pprpos ;
            features[index++] = psense_pprpos;
            String psense_pdepsubcat = pSense + " " + pdepsubcat;
            features[index++] = psense_pdepsubcat;
            String psense_pchilddepset = pSense + " " + pchilddepset;
            features[index++] = psense_pchilddepset;
            String psense_pchildposset = pSense + " " + pchildposset;
            features[index++] = psense_pchildposset;
            String psense_pchildwset = pSense + " " + pchildwset;
            features[index++] = psense_pchildwset;

            //pprw + ... (20 bits for pprw)
            int pprw_pprpos = (pprw<<10) | pprpos ;
            features[index++] = pprw_pprpos;
            String pprw_pdepsubcat = pprw + " " + pdepsubcat;
            features[index++] = pprw_pdepsubcat;
            String pprw_pchilddepset = pprw + " " + pchilddepset;
            features[index++] = pprw_pchilddepset;
            String pprw_pchildposset = pprw + " " + pchildposset;
            features[index++] = pprw_pchildposset;
            String pprw_pchildwset = pprw + " " + pchildwset;
            features[index++] = pprw_pchildwset;

            //pprpos + ...(10 bits for pprpos)
            String pprpos_pdepsubcat = pprpos + " " + pdepsubcat;
            features[index++] = pprpos_pdepsubcat;
            String pprpos_pchilddepset = pprpos + " " + pchilddepset;
            features[index++] = pprpos_pchilddepset;
            String pprpos_pchildposset = pprpos + " " + pchildposset;
            features[index++] = pprpos_pchildposset;
            String pprpos_pchildwset = pprpos + " " + pchildwset;
            features[index++] = pprpos_pchildwset;

            //pdepsubcat + ...
            String pdepsubcat_pchilddepset = pdepsubcat + " " + pchilddepset;
            features[index++] = pdepsubcat_pchilddepset;
            String pdepsubcat_pchildposset = pdepsubcat + " " + pchildposset;
            features[index++] = pdepsubcat_pchildposset;
            String pdepsubcat_pchildwset = pdepsubcat + " " + pchildwset;
            features[index++] = pdepsubcat_pchildwset;

            //pchilddepset + ...
            String pchilddepset_pchildposset = pchilddepset + " " + pchildposset;
            features[index++] = pchilddepset_pchildposset;
            String pchilddepset_pchildwset = pchilddepset + " " + pchildwset;
            features[index++] = pchilddepset_pchildwset;

            //pchildposset + ...
            String pchildposset_pchildwset = pchildposset + " " + pchildwset;
            features[index++] = pchildposset_pchildwset;

            //////////////////////////////////////////////////////////////////////
            ///////////////// ARGUMENT-ARGUMENT CONJOINED FEATURES ////////////
            //////////////////////////////////////////////////////////////////////
            //todo 20 bits for words, 10 bits for pos, 10 bits for dependencies
            // aw, apos, adeprel, deprelpath, pospath, position, leftw, leftpos, rightw, rightpos, leftsiblingw,
            // leftsiblingpos, rightsiblingw, rightsiblingpos

            int aw_apos = (aw<<10) | apos;
            features[index++] = aw_apos;
            int aw_adeprel = (aw<<10) | adeprel;
            features[index++] = aw_adeprel;
            String aw_deprelpath = aw+" "+ deprelpath;
            features[index++] = aw_deprelpath;
            String aw_pospath = aw+" "+ pospath;
            features[index++] = aw_pospath;
            int aw_position = (aw<<2) | position;
            features[index++] = aw_position;
            long aw_leftw = (aw<< 20) | leftw;
            features[index++] = aw_leftw;
            int aw_leftpos = (aw<<10) | leftpos;
            features[index++] = aw_leftpos;
            long aw_rightw = (aw<< 20) | rightw;
            features[index++] = aw_rightw;
            int aw_rightpos = (aw<<10) | rightpos;
            features[index++] = aw_rightpos;
            long aw_leftsiblingw = (aw<< 20) | leftsiblingw;
            features[index++] = aw_leftsiblingw;
            int aw_leftsiblingpos = (aw<<10) | leftsiblingpos;
            features[index++] = aw_leftsiblingpos;
            long aw_rightsiblingw = (aw<< 20) | rightsiblingw;
            features[index++] = aw_rightsiblingw;
            int aw_rightsiblingpos = (aw<<10) | rightsiblingpos;
            features[index++] = aw_rightsiblingpos;

            int apos_adeprel = (apos<<10) | adeprel;
            features[index++] = apos_adeprel;
            String apos_deprelpath = apos+" "+ deprelpath;
            features[index++] = apos_deprelpath;
            String apos_pospath = apos+" "+ pospath;
            features[index++] = apos_pospath;
            int apos_position = (apos<<2) | position;
            features[index++] = apos_position;
            int apos_leftw = (leftw<< 10) | apos;
            features[index++] = apos_leftw;
            int apos_leftpos = (apos<<10) | leftpos;
            features[index++] = apos_leftpos;
            int apos_rightw = (rightw<< 10) | apos;
            features[index++] = apos_rightw;
            int apos_rightpos = (apos<<10) | rightpos;
            features[index++] = apos_rightpos;
            int apos_leftsiblingw = (leftsiblingw<< 10) | apos;
            features[index++] = apos_leftsiblingw;
            int apos_leftsiblingpos = (apos<<10) | leftsiblingpos;
            features[index++] = apos_leftsiblingpos;
            int apos_rightsiblingw = (rightsiblingw<< 10) | apos;
            features[index++] = apos_rightsiblingw;
            int apos_rightsiblingpos = (apos<<10) | rightsiblingpos;
            features[index++] = apos_rightsiblingpos;

            String adeprel_deprelpath = adeprel+" "+ deprelpath;
            features[index++] = adeprel_deprelpath;
            String adeprel_pospath = adeprel+" "+ pospath;
            features[index++] = adeprel_pospath;
            int adeprel_position = (adeprel<<2) | position;
            features[index++] = adeprel_position;
            int adeprel_leftw = (leftw<< 10) | adeprel;
            features[index++] = adeprel_leftw;
            int adeprel_leftpos = (adeprel<<10) | leftpos;
            features[index++] = adeprel_leftpos;
            int adeprel_rightw = (rightw<< 10) | adeprel;
            features[index++] = adeprel_rightw;
            int adeprel_rightpos = (adeprel<<10) | rightpos;
            features[index++] = adeprel_rightpos;
            int adeprel_leftsiblingw = (leftsiblingw<< 10) | adeprel;
            features[index++] = adeprel_leftsiblingw;
            int adeprel_leftsiblingpos = (adeprel<<10) | leftsiblingpos;
            features[index++] = adeprel_leftsiblingpos;
            int adeprel_rightsiblingw = (rightsiblingw<< 10) | adeprel;
            features[index++] = adeprel_rightsiblingw;
            int adeprel_rightsiblingpos = (adeprel<<10) | rightsiblingpos;
            features[index++] = adeprel_rightsiblingpos;


            String deprelpath_pospath = deprelpath+" "+ pospath;
            features[index++] = deprelpath_pospath;
            String deprelpath_position = deprelpath + " " +position;
            features[index++] = deprelpath_position;
            String deprelpath_leftw = leftw +" "+ deprelpath;
            features[index++] = deprelpath_leftw;
            String deprelpath_leftpos = deprelpath + " " +leftpos;
            features[index++] = deprelpath_leftpos;
            String deprelpath_rightw = rightw + " "+deprelpath;
            features[index++] = deprelpath_rightw;
            String deprelpath_rightpos = deprelpath +" " +rightpos;
            features[index++] = deprelpath_rightpos;
            String deprelpath_leftsiblingw = leftsiblingw+" "+deprelpath;
            features[index++] = deprelpath_leftsiblingw;
            String deprelpath_leftsiblingpos = deprelpath +" "+ leftsiblingpos;
            features[index++] = deprelpath_leftsiblingpos;
            String deprelpath_rightsiblingw = rightsiblingw+" "+deprelpath;
            features[index++] = deprelpath_rightsiblingw;
            String deprelpath_rightsiblingpos = deprelpath+" "+rightsiblingpos;
            features[index++] = deprelpath_rightsiblingpos;


            String pospath_position = pospath + " " +position;
            features[index++] = pospath_position;
            String pospath_leftw = leftw +" "+ pospath;
            features[index++] = pospath_leftw;
            String pospath_leftpos = pospath + " " +leftpos;
            features[index++] = pospath_leftpos;
            String pospath_rightw = rightw + " "+pospath;
            features[index++] = pospath_rightw;
            String pospath_rightpos = pospath +" " +rightpos;
            features[index++] = pospath_rightpos;
            String pospath_leftsiblingw = leftsiblingw+" "+pospath;
            features[index++] = pospath_leftsiblingw;
            String pospath_leftsiblingpos = pospath +" "+ leftsiblingpos;
            features[index++] = pospath_leftsiblingpos;
            String pospath_rightsiblingw = rightsiblingw+" "+pospath;
            features[index++] = pospath_rightsiblingw;
            String pospath_rightsiblingpos = pospath+" "+rightsiblingpos;
            features[index++] = pospath_rightsiblingpos;

            int position_leftw = (leftw<< 2) | position;
            features[index++] = position_leftw;
            int position_leftpos = (leftpos<<2) | position;
            features[index++] = position_leftpos;
            int position_rightw = (rightw<< 2) | position;
            features[index++] = position_rightw;
            int position_rightpos = (rightpos<<2) | position;
            features[index++] = position_rightpos;
            int position_leftsiblingw = (leftsiblingw<< 2) | position;
            features[index++] = position_leftsiblingw;
            int position_leftsiblingpos = (leftsiblingpos<<2) | position;
            features[index++] = position_leftsiblingpos;
            int position_rightsiblingw = (rightsiblingw<< 2) | position;
            features[index++] = position_rightsiblingw;
            int position_rightsiblingpos = (rightsiblingpos<<2) | position;
            features[index++] = position_rightsiblingpos;

            int leftw_leftpos = (leftw<<10) | leftpos;
            features[index++] = leftw_leftpos;
            long leftw_rightw = (leftw<< 20) | rightw;
            features[index++] = leftw_rightw;
            int leftw_rightpos = (leftw<<10) | rightpos;
            features[index++] = leftw_rightpos;
            long leftw_leftsiblingw = (leftw<< 20) | leftsiblingw;
            features[index++] = leftw_leftsiblingw;
            int leftw_leftsiblingpos = (leftw<<10) | leftsiblingpos;
            features[index++] = leftw_leftsiblingpos;
            long leftw_rightsiblingw = (leftw<< 20) | rightsiblingw;
            features[index++] = leftw_rightsiblingw;
            int leftw_rightsiblingpos = (leftw<<10) | rightsiblingpos;
            features[index++] = leftw_rightsiblingpos;

            int leftpos_rightw = (rightw<< 10) | leftpos;
            features[index++] = leftpos_rightw;
            int leftpos_rightpos = (leftpos<<10) | rightpos;
            features[index++] = leftpos_rightpos;
            int leftpos_leftsiblingw = (leftsiblingw<< 10) | leftpos;
            features[index++] = leftpos_leftsiblingw;
            int leftpos_leftsiblingpos = (leftpos<<10) | leftsiblingpos;
            features[index++] = leftpos_leftsiblingpos;
            int leftpos_rightsiblingw = (rightsiblingw<< 10) | leftpos;
            features[index++] = leftpos_rightsiblingw;
            int leftpos_rightsiblingpos = (leftpos<<10) | rightsiblingpos;
            features[index++] = leftpos_rightsiblingpos;

            int rightw_rightpos = (rightw<<10) | rightpos;
            features[index++] = rightw_rightpos;
            long rightw_leftsiblingw = (rightw<< 20) | leftsiblingw;
            features[index++] = rightw_leftsiblingw;
            int rightw_leftsiblingpos = (rightw<<10) | leftsiblingpos;
            features[index++] = rightw_leftsiblingpos;
            long rightw_rightsiblingw = (rightw<< 20) | rightsiblingw;
            features[index++] = rightw_rightsiblingw;
            int rightw_rightsiblingpos = (rightw<<10) | rightsiblingpos;
            features[index++] = rightw_rightsiblingpos;


            int rightpos_leftsiblingw = (leftsiblingw<< 10) | rightpos;
            features[index++] = rightpos_leftsiblingw;
            int rightpos_leftsiblingpos = (rightpos<<10) | leftsiblingpos;
            features[index++] = rightpos_leftsiblingpos;
            int rightpos_rightsiblingw = (rightsiblingw<< 10) | rightpos;
            features[index++] = rightpos_rightsiblingw;
            int rightpos_rightsiblingpos = (rightpos<<10) | rightsiblingpos;
            features[index++] = rightpos_rightsiblingpos;


            int leftsiblingw_leftsiblingpos = (leftsiblingw<<10) | leftsiblingpos;
            features[index++] = leftsiblingw_leftsiblingpos;
            long leftsiblingw_rightsiblingw = (leftsiblingw<< 20) | rightsiblingw;
            features[index++] = leftsiblingw_rightsiblingw;
            int leftsiblingw_rightsiblingpos = (leftsiblingw<<10) | rightsiblingpos;
            features[index++] = leftsiblingw_rightsiblingpos;

            long leftsiblingpos_rightsiblingw = (rightsiblingw<< 10) | leftsiblingpos;
            features[index++] = leftsiblingpos_rightsiblingw;
            int leftsiblingpos_rightsiblingpos = (rightsiblingpos<<10) | leftsiblingpos;
            features[index++] = leftsiblingpos_rightsiblingpos;

            int rightSiblingw_rightSiblingpos = (rightsiblingw<<10) | rightsiblingpos;
            features[index++] = rightSiblingw_rightSiblingpos;

             */
            //////////////////////////////////////////////////////////////////////
            ///////////////// PREDICATE-ARGUMENT CONJOINED FEATURES //////////////
            //////////////////////////////////////////////////////////////////////

            // todo 20 bits for words, 10 bits for pos, 10 bits for dependencies
            // todo e.g. (pw<<20) | aw ==> 20+20> 32 ==> long
            // pw + argument features
            long pw_aw = (pw<<20) | aw;
            features[index++] = pw_aw;
            // todo (pw<<10) | pos ==> 10+20<32 ==> int
            int pw_apos = (pw<<10) | apos ;
            features[index++] = pw_apos;
            int pw_adeprel = (pw<<10) | adeprel;
            features[index++] = pw_adeprel;
            //StringBuilder pw_deprelpath= new StringBuilder();
            //pw_deprelpath.append(pw);
            //pw_deprelpath.append(" ");
            //pw_deprelpath.append(deprelpath);
            String pw_deprelpath = pw +" "+deprelpath;
            features[index++] = pw_deprelpath;
            //StringBuilder pw_pospath= new StringBuilder();
            //pw_deprelpath.append(pw);
            //pw_deprelpath.append(" ");
            //pw_deprelpath.append(pospath);
            String pw_pospath = pw +" "+pospath;
            features[index++] = pw_pospath;
            int pw_position = (pw<<2) | position;
            features[index++] = pw_position;
            long pw_leftw = (pw<<20) | leftw;
            features[index++] = pw_leftw;
            int pw_leftpos = (pw<<10) | leftpos;
            features[index++] = pw_leftpos;
            long pw_rightw = (pw<<20) | rightw;
            features[index++] = pw_rightw;
            int pw_rightpos = (pw<<10) | rightpos;
            features[index++] = pw_rightpos;
            long pw_leftsiblingw = (pw<<20) | leftsiblingw;
            features[index++] = pw_leftsiblingw;
            int pw_leftsiblingpos= (pw<<10) | leftsiblingpos;
            features[index++] = pw_leftsiblingpos;
            long pw_rightsiblingw = (pw<<20) | rightsiblingw;
            features[index++] = pw_rightsiblingw;
            int pw_rightsiblingpos = (pw<<10) | rightsiblingpos;
            features[index++] = pw_rightsiblingpos;

            //ppos + argument features
            int aw_ppos = (aw<<10) | ppos;
            features[index++] = aw_ppos;
            int ppos_apos = (ppos<<10) | apos;
            features[index++] = ppos_apos;
            int ppos_adeprel = (ppos<<10) | adeprel;
            features[index++] = ppos_adeprel;
            //StringBuilder ppos_deprelpath = new StringBuilder();
            //ppos_deprelpath.append(ppos);
            //ppos_deprelpath.append(" ");
            //ppos_deprelpath.append(deprelpath);
            String ppos_deprelpath = ppos+" "+deprelpath;
            features[index++] = ppos_deprelpath;
            //StringBuilder ppos_pospath = new StringBuilder();
            //ppos_pospath.append(ppos);
            //ppos_pospath.append(" ");
            //ppos_pospath.append(pospath);
            String ppos_pospath= ppos +" "+pospath;
            features[index++] = ppos_pospath;
            int ppos_position = (ppos<<2) | position;
            features[index++] = ppos_position;
            int leftw_ppos = (leftw<<10) | ppos;
            features[index++] = leftw_ppos;
            int ppos_leftpos = (ppos<<10) | leftpos;
            features[index++] = ppos_leftpos;
            int rightw_ppos = (rightw<<10) | ppos;
            features[index++] = rightw_ppos;
            int ppos_rightpos = (ppos<<10) | rightpos;
            features[index++] = ppos_rightpos;
            int leftsiblingw_ppos = (leftsiblingw<<10) | ppos;
            features[index++] = leftsiblingw_ppos;
            int ppos_leftsiblingpos = (ppos<<10) | leftsiblingpos;
            features[index++] = ppos_leftsiblingpos;
            int rightsiblingw_ppos = (rightsiblingw<<10) | ppos;
            features[index++] = rightsiblingw_ppos;
            int ppos_rightsiblingpos = (ppos<<10) | rightsiblingpos;
            features[index++] = ppos_rightsiblingpos;


            //pdeprel + argument features
            int aw_pdeprel = (aw<<10) | pdeprel;
            features[index++] = aw_pdeprel;
            int pdeprel_apos = (pdeprel<<10) | apos;
            features[index++] = pdeprel_apos;
            int pdeprel_adeprel = (pdeprel<<10) | adeprel;
            features[index++] = pdeprel_adeprel;
            //StringBuilder pdeprel_deprelpath = new StringBuilder();
            //pdeprel_deprelpath.append(pdeprel);
            //pdeprel_deprelpath.append(" ");
            //pdeprel_deprelpath.append(deprelpath);
            String pdeprel_deprelpath = pdeprel + " "+deprelpath;
            features[index++] = pdeprel_deprelpath;
            //StringBuilder pdeprel_pospath= new StringBuilder();
            //pdeprel_pospath.append(pdeprel);
            //pdeprel_pospath.append(" ");
            //pdeprel_pospath.append(pospath);
            String pdeprel_pospath = pdeprel +" "+pospath;
            features[index++] = pdeprel_pospath;
            int pdeprel_position = (pdeprel<<2) | position;
            features[index++] = pdeprel_position;
            int leftw_pdeprel = (leftw<<10) | pdeprel;
            features[index++] = leftw_pdeprel;
            int pdeprel_leftpos = (pdeprel<<10) | leftpos;
            features[index++] = pdeprel_leftpos;
            int rightw_pdeprel = (rightw<<10) | pdeprel;
            features[index++] = rightw_pdeprel;
            int pdeprel_rightpos = (pdeprel<<10) | rightpos;
            features[index++] = pdeprel_rightpos;
            int leftsiblingw_pdeprel = (leftsiblingw<<10) |pdeprel;
            features[index++] = leftsiblingw_pdeprel;
            int pdeprel_leftsiblingpos = (pdeprel<<10) | leftsiblingpos;
            features[index++] = pdeprel_leftsiblingpos;
            int rightsiblingw_pdeprel = (rightsiblingw<<10) | pdeprel;
            features[index++] = rightsiblingw_pdeprel;
            int pdeprel_rightsiblingpos = (pdeprel<<10) | rightsiblingpos;
            features[index++] = pdeprel_rightsiblingpos;


            //plem + argument features
            long aw_plem = (aw<<20) | plem;
            features[index++] = aw_plem;
            int plem_apos = (plem<<10) | apos;
            features[index++] = plem_apos;
            int plem_adeprel = (plem<<10) | adeprel;
            features[index++] = plem_adeprel;
            //StringBuilder plem_deprelpath = new StringBuilder();
            //plem_deprelpath.append(plem);
            //plem_deprelpath.append(" ");
            //plem_deprelpath.append(deprelpath);
            String plem_deprelpath = plem +" "+deprelpath;
            features[index++] = plem_deprelpath;
            //StringBuilder plem_pospath= new StringBuilder();
            //plem_pospath.append(plem);
            //plem_pospath.append(" ");
            //plem_pospath.append(pospath);
            String plem_pospath = plem +" "+pospath;
            features[index++] = plem_pospath;
            int plem_position = (plem<<2) | position;
            features[index++] = plem_position;
            long leftw_plem = (leftw<<20) | plem;
            features[index++] = leftw_plem;
            int plem_leftpos = (plem<<10) | leftpos;
            features[index++] = plem_leftpos;
            long rightw_plem = (rightw<<20) | plem;
            features[index++] = rightw_plem;
            int plem_rightpos = (plem<<10) | rightpos;
            features[index++] = plem_rightpos;
            long leftsiblingw_plem = (leftsiblingw<<20) |plem;
            features[index++] = leftsiblingw_plem;
            int plem_leftsiblingpos = (plem<<10) | leftsiblingpos;
            features[index++] = plem_leftsiblingpos;
            long rightsiblingw_plem = (rightsiblingw<<20) | plem;
            features[index++] = rightsiblingw_plem;
            int plem_rightsiblingpos = (plem<<10) | rightsiblingpos;
            features[index++] = plem_rightsiblingpos;

            //psense + argument features
            //StringBuilder psense_aw = new StringBuilder();
            //psense_aw.append(psense);
            //psense_aw.append(" ");
            //psense_aw.append(aw);
            String psense_aw = pSense + " " + aw;
            features[index++] = psense_aw;
            //StringBuilder psense_apos = new StringBuilder();
            //psense_apos.append(psense);
            //psense_apos.append(" ");
            //psense_apos.append(apos);
            String psense_apos = pSense +" "+apos;
            features[index++] = psense_apos;
            //StringBuilder psense_adeprel = new StringBuilder();
            //psense_adeprel.append(psense);
            //psense_adeprel.append(" ");
            //psense_adeprel.append(adeprel);
            String psense_adeprel = pSense + " "+ adeprel;
            features[index++] = psense_adeprel;
            //StringBuilder psense_deprelpath = new StringBuilder();
            //psense_deprelpath.append(psense);
            //psense_deprelpath.append(" ");
            //psense_deprelpath.append(deprelpath);
            String psense_deprelpath = pSense + " "+ deprelpath;
            features[index++] = psense_deprelpath;
            //StringBuilder psense_pospath = new StringBuilder();
            //psense_pospath.append(psense);
            //psense_pospath.append(" ");
            //psense_pospath.append(pospath);
            String psense_pospath = pSense + " "+ pospath;
            features[index++] = psense_pospath;
            //StringBuilder psense_position = new StringBuilder();
            //psense_position.append(psense);
            //psense_position.append(" ");
            //psense_position.append(position);
            String psense_position = pSense + " " +position;
            features[index++] = psense_position;
            //StringBuilder psense_leftw = new StringBuilder();
            //psense_leftw.append(psense);
            //psense_leftw.append(" ");
            //psense_leftw.append(leftw);
            String psense_leftw = pSense + " " + leftw;
            features[index++] = psense_leftw;
            //StringBuilder psense_leftpos = new StringBuilder();
            //psense_leftpos.append(psense);
            //psense_leftpos.append(" ");
            //psense_leftpos.append(leftpos);
            String psense_leftpos = pSense + " " + leftpos;
            features[index++] = psense_leftpos;
            //StringBuilder psense_rightw = new StringBuilder();
            //psense_rightw.append(psense);
            //psense_rightw.append(" ");
            //psense_rightw.append(rightw);
            String psense_rightw = pSense + " " + rightw;
            features[index++] = psense_rightw;
            //StringBuilder psense_rightpos = new StringBuilder();
            //psense_rightpos.append(psense);
            //psense_rightpos.append(" ");
            //psense_rightpos.append(rightpos);
            String psense_rightpos = pSense + " " + rightpos;
            features[index++] = psense_rightpos;
            //StringBuilder psense_leftsiblingw = new StringBuilder();
            //psense_leftsiblingw.append(psense);
            //psense_leftsiblingw.append(" ");
            //psense_leftsiblingw.append(leftsiblingw);
            String psense_leftsiblingw = pSense +" " + leftsiblingw;
            features[index++] = psense_leftsiblingw;
            //StringBuilder psense_leftsiblingpos = new StringBuilder();
            //psense_leftsiblingpos.append(psense);
            //psense_leftsiblingpos.append(" ");
            //psense_leftsiblingpos.append(leftsiblingpos);
            String psense_leftsiblingpos = pSense + " " + leftsiblingpos;
            features[index++] = psense_leftsiblingpos;
            //StringBuilder psense_rightsiblingw = new StringBuilder();
            //psense_rightsiblingw.append(psense);
            //psense_rightsiblingw.append(" ");
            //psense_rightsiblingw.append(rightsiblingw);
            String psense_rightsiblingw = pSense +" "+rightsiblingw;
            features[index++] = psense_rightsiblingw;
            //StringBuilder psense_rightsiblingpos = new StringBuilder();
            //psense_rightsiblingpos.append(psense);
            //psense_rightsiblingpos.append(psense);
            //psense_rightsiblingpos.append(" ");
            //psense_rightsiblingpos.append(rightsiblingpos);
            String psense_rightsiblingpos = pSense +" "+rightsiblingpos;
            features[index++] = psense_rightsiblingpos;


            //pprw  + argument features
            long aw_pprw = (aw<<20) | pprw;
            features[index++] = aw_pprw;
            int pprw_apos = (pprw<<10) | apos;
            features[index++] = pprw_apos;
            int pprw_adeprel = (pprw<<10) | adeprel;
            features[index++] = pprw_adeprel;
            //StringBuilder pprw_deprelpath = new StringBuilder();
            //pprw_deprelpath.append(pprw);
            //pprw_deprelpath.append(" ");
            //pprw_deprelpath.append(deprelpath);
            String pprw_deprelpath = pprw +" "+deprelpath;
            features[index++] = pprw_deprelpath;
            //StringBuilder pprw_pospath= new StringBuilder();
            //pprw_pospath.append(pprw);
            //pprw_pospath.append(" ");
            //pprw_pospath.append(pospath);
            String pprw_pospath = pprw +" "+ pospath;
            features[index++] = pprw_pospath;
            int pprw_position = (pprw<<2) | position;
            features[index++] = pprw_position;
            long leftw_pprw = (leftw<<20) | pprw;
            features[index++] = leftw_pprw;
            int pprw_leftpos = (pprw<<10) | leftpos;
            features[index++] = pprw_leftpos;
            long rightw_pprw = (rightw<<20) | pprw;
            features[index++] = rightw_pprw;
            int pprw_rightpos = (pprw<<10) | rightpos;
            features[index++] = pprw_rightpos;
            long leftsiblingw_pprw = (leftsiblingw<<20) |pprw;
            features[index++] = leftsiblingw_pprw;
            int pprw_leftsiblingpos = (pprw<<10) | leftsiblingpos;
            features[index++] = pprw_leftsiblingpos;
            long rightsiblingw_pprw = (rightsiblingw<<20) | pprw;
            features[index++] = rightsiblingw_pprw;
            int pprw_rightsiblingpos = (pprw<<10) | rightsiblingpos;
            features[index++] = pprw_rightsiblingpos;


            //pdeprel + argument features
            int aw_pprpos = (aw<<10) | pprpos;
            features[index++] = aw_pprpos;
            int pprpos_apos = (pprpos<<10) | apos;
            features[index++] = pprpos_apos;
            int pprpos_adeprel = (pprpos<<10) | adeprel;
            features[index++] = pprpos_adeprel;
            //StringBuilder pprpos_deprelpath = new StringBuilder();
            //pprpos_deprelpath.append(pprpos);
            //pprpos_deprelpath.append(" ");
            //pprpos_deprelpath.append(deprelpath);
            String pprpos_deprelpath = pprpos +" "+deprelpath;
            features[index++] = pprpos_deprelpath;
            //StringBuilder pprpos_pospath= new StringBuilder();
            //pprpos_pospath.append(pprpos);
            //pprpos_pospath.append(" ");
            //pprpos_pospath.append(pospath);
            String pprpos_pospath= pprpos +" "+pospath;
            features[index++] = pprpos_pospath;
            int pprpos_position = (pprpos<<2) | position;
            features[index++] = pprpos_position;
            int leftw_pprpos = (leftw<<10) | pprpos;
            features[index++] = leftw_pprpos;
            int pprpos_leftpos = (pprpos<<10) | leftpos;
            features[index++] = pprpos_leftpos;
            int rightw_pprpos = (rightw<<10) | pprpos;
            features[index++] = rightw_pprpos;
            int pprpos_rightpos = (pprpos<<10) | rightpos;
            features[index++] = pprpos_rightpos;
            int leftsiblingw_pprpos = (leftsiblingw<<10) |pprpos;
            features[index++] = leftsiblingw_pprpos;
            int pprpos_leftsiblingpos = (pprpos<<10) | leftsiblingpos;
            features[index++] = pprpos_leftsiblingpos;
            int rightsiblingw_pprpos = (rightsiblingw<<10) | pprpos;
            features[index++] = rightsiblingw_pprpos;
            int pprpos_rightsiblingpos = (pprpos<<10) | rightsiblingpos;
            features[index++] = pprpos_rightsiblingpos;

            //pchilddepset + argument features
            //StringBuilder pchilddepset_aw = new StringBuilder();
            //pchilddepset_aw.append(pchilddepset);
            //pchilddepset_aw.append(" ");
            //pchilddepset_aw.append(aw);
            String pchilddepset_aw = pchilddepset +" "+aw;
            features[index++] = pchilddepset_aw;
            //StringBuilder pchilddepset_apos = new StringBuilder();
            //pchilddepset_apos.append(pchilddepset);
            //pchilddepset_apos.append(" ");
            //pchilddepset_apos.append(apos);
            String pchilddepset_apos = pchilddepset +" "+ apos;
            features[index++] = pchilddepset_apos;
            //StringBuilder pchilddepset_adeprel = new StringBuilder();
            //pchilddepset_adeprel.append(pchilddepset);
            //pchilddepset_adeprel.append(" ");
            //pchilddepset_adeprel.append(adeprel);
            String pchilddepset_adeprel = pchilddepset +" "+ adeprel;
            features[index++] = pchilddepset_adeprel;
            //StringBuilder pchilddepset_deprelpath = new StringBuilder();
            //pchilddepset_deprelpath.append(pchilddepset);
            //pchilddepset_deprelpath.append(" ");
            //pchilddepset_deprelpath.append(deprelpath);
            String pchilddepset_deprelpath = pchilddepset +" "+ deprelpath;
            features[index++] = pchilddepset_deprelpath;
            //StringBuilder pchilddepset_pospath = new StringBuilder();
            //pchilddepset_pospath.append(pchilddepset);
            //pchilddepset_pospath.append(" ");
            //pchilddepset_pospath.append(pospath);
            String pchilddepset_pospath = pchilddepset +" "+pospath;
            features[index++] = pchilddepset_pospath;
            //StringBuilder pchilddepset_position = new StringBuilder();
            //pchilddepset_position.append(pchilddepset);
            //pchilddepset_position.append(" ");
            //pchilddepset_position.append(position);
            String pchilddepset_position = pchilddepset +" "+position;
            features[index++] = pchilddepset_position;
            //StringBuilder pchilddepset_leftw = new StringBuilder();
            //pchilddepset_leftw.append(pchilddepset);
            //pchilddepset_leftw.append(" ");
            //pchilddepset_leftw.append(leftw);
            String pchilddepset_leftw = pchilddepset +" "+ leftw;
            features[index++] = pchilddepset_leftw;
            //StringBuilder pchilddepset_leftpos = new StringBuilder();
            //pchilddepset_leftpos.append(pchilddepset);
            //pchilddepset_leftpos.append(" ");
            //pchilddepset_leftpos.append(leftpos);
            String pchilddepset_leftpos = pchilddepset +" "+leftpos;
            features[index++] = pchilddepset_leftpos;
            //StringBuilder pchilddepset_rightw = new StringBuilder();
            //pchilddepset_rightw.append(pchilddepset);
            //pchilddepset_rightw.append(" ");
            //pchilddepset_rightw.append(rightw);
            String pchilddepset_rightw = pchilddepset +" "+rightw;
            features[index++] = pchilddepset_rightw;
            //StringBuilder pchilddepset_rightpos = new StringBuilder();
            //pchilddepset_rightpos.append(pchilddepset);
            //pchilddepset_rightpos.append(" ");
            //pchilddepset_rightpos.append(rightpos);
            String pchilddepset_rightpos = pchilddepset +" "+rightpos;
            features[index++] = pchilddepset_rightpos;
            //StringBuilder pchilddepset_leftsiblingw = new StringBuilder();
            //pchilddepset_leftsiblingw.append(pchilddepset);
            //pchilddepset_leftsiblingw.append(" ");
            //pchilddepset_leftsiblingw.append(leftsiblingw);
            String pchilddepset_leftsiblingw = pchilddepset +" "+leftsiblingw;
            features[index++] = pchilddepset_leftsiblingw;
            //StringBuilder pchilddepset_leftsiblingpos = new StringBuilder();
            //pchilddepset_leftsiblingpos.append(pchilddepset);
            //pchilddepset_leftsiblingpos.append(" ");
            //pchilddepset_leftsiblingpos.append(leftsiblingpos);
            String pchilddepset_leftsiblingpos = pchilddepset +" "+ leftsiblingpos;
            features[index++] = pchilddepset_leftsiblingpos;
            //StringBuilder pchilddepset_rightsiblingw = new StringBuilder();
            //pchilddepset_rightsiblingw.append(pchilddepset);
            //pchilddepset_rightsiblingw.append(" ");
            //pchilddepset_rightsiblingw.append(rightsiblingw);
            String pchilddepset_rightsiblingw = pchilddepset +" "+rightsiblingw;
            features[index++] = pchilddepset_rightsiblingw;
            //StringBuilder pchilddepset_rightsiblingpos = new StringBuilder();
            //pchilddepset_rightsiblingpos.append(pchilddepset);
            //pchilddepset_rightsiblingpos.append(" ");
            //pchilddepset_rightsiblingpos.append(rightsiblingpos);
            String pchilddepset_rightsiblingpos = pchilddepset + " "+rightsiblingpos;
            features[index++] = pchilddepset_rightsiblingpos;


            //pdepsubcat + argument features
            //StringBuilder pdepsubcat_aw = new StringBuilder();
            //pdepsubcat_aw.append(pdepsubcat);
            //pdepsubcat_aw.append(" ");
            //pdepsubcat_aw.append(aw);
            String pdepsubcat_aw =pdepsubcat +" " + aw;
            features[index++] = pdepsubcat_aw;
            //StringBuilder pdepsubcat_apos = new StringBuilder();
            //pdepsubcat_apos.append(pdepsubcat);
            //pdepsubcat_apos.append(" ");
            //pdepsubcat_apos.append(apos);
            String pdepsubcat_apos = pdepsubcat +" " + apos;
            features[index++] = pdepsubcat_apos;
            //StringBuilder pdepsubcat_adeprel = new StringBuilder();
            //pdepsubcat_adeprel.append(pdepsubcat);
            //pdepsubcat_adeprel.append(" ");
            //pdepsubcat_adeprel.append(adeprel);
            String pdepsubcat_adeprel = pdepsubcat +" "+adeprel;
            features[index++] = pdepsubcat_adeprel;
            //StringBuilder pdepsubcat_deprelpath = new StringBuilder();
            //pdepsubcat_deprelpath.append(pdepsubcat);
            //pdepsubcat_deprelpath.append(" ");
            //pdepsubcat_deprelpath.append(deprelpath);
            String pdepsubcat_deprelpath = pdepsubcat +" "+deprelpath;
            features[index++] = pdepsubcat_deprelpath;
            //StringBuilder pdepsubcat_pospath = new StringBuilder();
            //pdepsubcat_pospath.append(pdepsubcat);
            //pdepsubcat_pospath.append(" ");
            //pdepsubcat_pospath.append(pospath);
            String pdepsubcat_pospath = pdepsubcat +" "+pospath;
            features[index++] = pdepsubcat_pospath;
            //StringBuilder pdepsubcat_position = new StringBuilder();
            //pdepsubcat_position.append(pdepsubcat);
            //pdepsubcat_position.append(" ");
            //pdepsubcat_position.append(position);
            String pdepsubcat_position = pdepsubcat +" "+position;
            features[index++] = pdepsubcat_position;
            //StringBuilder pdepsubcat_leftw = new StringBuilder();
            //pdepsubcat_leftw.append(pdepsubcat);
            //pdepsubcat_leftw.append(" ");
            //pdepsubcat_leftw.append(leftw);
            String pdepsubcat_leftw = pdepsubcat +" "+leftw;
            features[index++] = pdepsubcat_leftw;
            //StringBuilder pdepsubcat_leftpos = new StringBuilder();
            //pdepsubcat_leftpos.append(pdepsubcat);
            //pdepsubcat_leftpos.append(" ");
            //pdepsubcat_leftpos.append(leftpos);
            String pdepsubcat_leftpos =pdepsubcat +" "+ leftpos;
            features[index++] = pdepsubcat_leftpos;
            //StringBuilder pdepsubcat_rightw = new StringBuilder();
            //pdepsubcat_rightw.append(pdepsubcat);
            //pdepsubcat_rightw.append(" ");
            //pdepsubcat_rightw.append(rightw);
            String pdepsubcat_rightw = pdepsubcat +" "+ rightw;
            features[index++] = pdepsubcat_rightw;
            //StringBuilder pdepsubcat_rightpos = new StringBuilder();
            //pdepsubcat_rightpos.append(pdepsubcat);
            //pdepsubcat_rightpos.append(" ");
            //pdepsubcat_rightpos.append(rightpos);
            String pdepsubcat_rightpos = pdepsubcat +" "+rightpos;
            features[index++] = pdepsubcat_rightpos;
            //StringBuilder pdepsubcat_leftsiblingw = new StringBuilder();
            //pdepsubcat_leftsiblingw.append(pdepsubcat);
            //pdepsubcat_leftsiblingw.append(" ");
            //pdepsubcat_leftsiblingw.append(leftsiblingw);
            String pdepsubcat_leftsiblingw =pdepsubcat +" "+ leftsiblingw;
            features[index++] = pdepsubcat_leftsiblingw;
            //StringBuilder pdepsubcat_leftsiblingpos = new StringBuilder();
            //pdepsubcat_leftsiblingpos.append(pdepsubcat);
            //pdepsubcat_leftsiblingpos.append(" ");
            //pdepsubcat_leftsiblingpos.append(leftsiblingpos);
            String pdepsubcat_leftsiblingpos = pdepsubcat + " "+leftsiblingpos;
            features[index++] = pdepsubcat_leftsiblingpos;
            //StringBuilder pdepsubcat_rightsiblingw = new StringBuilder();
            //pdepsubcat_rightsiblingw.append(pdepsubcat);
            //pdepsubcat_rightsiblingw.append(" ");
            //pdepsubcat_rightsiblingw.append(rightsiblingw);
            String pdepsubcat_rightsiblingw = pdepsubcat +" "+rightsiblingw;
            features[index++] = pdepsubcat_rightsiblingw;
            //StringBuilder pdepsubcat_rightsiblingpos = new StringBuilder();
            //pdepsubcat_rightsiblingpos.append(pdepsubcat);
            //pdepsubcat_rightsiblingpos.append(" ");
            //pdepsubcat_rightsiblingpos.append(rightsiblingpos);
            String pdepsubcat_rightsiblingpos = pdepsubcat +" "+rightsiblingpos;
            features[index++] = pdepsubcat_rightsiblingpos;


            //pchildposset + argument features
            //StringBuilder pchildposset_aw = new StringBuilder();
            ///pchildposset_aw.append(pchildposset);
            //pchildposset_aw.append(" ");
            //pchildposset_aw.append(aw);
            String pchildposset_aw = pchildposset + " "+ aw;
            features[index++] = pchildposset_aw;
            //StringBuilder pchildposset_apos = new StringBuilder();
            //pchildposset_apos.append(pchildposset);
            //pchildposset_apos.append(" ");
            //pchildposset_apos.append(apos);
            String pchildposset_apos = pchildposset +" "+apos;
            features[index++] = pchildposset_apos;
            //StringBuilder pchildposset_adeprel = new StringBuilder();
            //pchildposset_adeprel.append(pchildposset);
            //pchildposset_adeprel.append(" ");
            //pchildposset_adeprel.append(adeprel);
            String pchildposset_adeprel = pchildposset + " " + adeprel;
            features[index++] = pchildposset_adeprel;
            //StringBuilder pchildposset_deprelpath = new StringBuilder();
            //pchildposset_deprelpath.append(pchildposset);
            //pchildposset_deprelpath.append(" ");
            //pchildposset_deprelpath.append(deprelpath);
            String pchildposset_deprelpath =pchildposset +" "+deprelpath;
            features[index++] = pchildposset_deprelpath;
            //StringBuilder pchildposset_pospath = new StringBuilder();
            //pchildposset_pospath.append(pchildposset);
            //pchildposset_pospath.append(" ");
            //pchildposset_pospath.append(pospath);
            String pchildposset_pospath = pchildposset +" "+ pospath;
            features[index++] = pchildposset_pospath;
            //StringBuilder pchildposset_position = new StringBuilder();
            //pchildposset_position.append(pchildposset);
            //pchildposset_position.append(" ");
            //pchildposset_position.append(position);
            String pchildposset_position = pchildposset + " "+ position;
            features[index++] = pchildposset_position;
            //StringBuilder pchildposset_leftw = new StringBuilder();
            //pchildposset_leftw.append(pchildposset);
            //pchildposset_leftw.append(" ");
            //pchildposset_leftw.append(leftw);
            String pchildposset_leftw = pchildposset +" "+leftw;
            features[index++] = pchildposset_leftw;
            //StringBuilder pchildposset_leftpos = new StringBuilder();
            //pchildposset_leftpos.append(pchildposset);
            //pchildposset_leftpos.append(" ");
            //pchildposset_leftpos.append(leftpos);
            String pchildposset_leftpos = pchildposset +" "+ leftpos;
            features[index++] = pchildposset_leftpos;
            //StringBuilder pchildposset_rightw = new StringBuilder();
            //pchildposset_rightw.append(pchildposset);
            //pchildposset_rightw.append(" ");
            //pchildposset_rightw.append(rightw);
            String pchildposset_rightw = pchildposset + " "+rightw;
            features[index++] = pchildposset_rightw;
            //StringBuilder pchildposset_rightpos = new StringBuilder();
            //pchildposset_rightpos.append(pchildposset);
            //pchildposset_rightpos.append(" ");
            //pchildposset_rightpos.append(rightpos);
            String pchildposset_rightpos = pchildposset + " "+rightpos;
            features[index++] = pchildposset_rightpos;
            //StringBuilder pchildposset_leftsiblingw = new StringBuilder();
            //pchildposset_leftsiblingw.append(pchildposset);
            //pchildposset_leftsiblingw.append(" ");
            //pchildposset_leftsiblingw.append(leftsiblingw);
            String pchildposset_leftsiblingw = pchildposset + " "+ leftsiblingw;
            features[index++] = pchildposset_leftsiblingw;
            //StringBuilder pchildposset_leftsiblingpos = new StringBuilder();
            //pchildposset_leftsiblingpos.append(pchildposset);
            //pchildposset_leftsiblingpos.append(" ");
            //pchildposset_leftsiblingpos.append(leftsiblingpos);
            String pchildposset_leftsiblingpos = pchildposset + " "+ leftsiblingpos;
            features[index++] = pchildposset_leftsiblingpos;
            //StringBuilder pchildposset_rightsiblingw = new StringBuilder();
            //pchildposset_rightsiblingw.append(pchildposset);
            //pchildposset_rightsiblingw.append(" ");
            //pchildposset_rightsiblingw.append(rightsiblingw);
            String pchildposset_rightsiblingw = pchildposset +" " +rightsiblingw;
            features[index++] = pchildposset_rightsiblingw;
            //StringBuilder pchildposset_rightsiblingpos = new StringBuilder();
            //pchildposset_rightsiblingpos.append(pchildposset);
            //pchildposset_rightsiblingpos.append(" ");
            //pchildposset_rightsiblingpos.append(rightsiblingpos);
            String pchildposset_rightsiblingpos = pchildposset +" "+rightsiblingpos;
            features[index++] = pchildposset_rightsiblingpos;


            //pchildwset + argument features
            //StringBuilder pchildwset_aw = new StringBuilder();
            //pchildwset_aw.append(pchildwset);
            //pchildwset_aw.append(" ");
            //pchildwset_aw.append(aw);
            String pchildwset_aw = pchildwset +" "+ aw;
            features[index++] = pchildwset_aw;
            //StringBuilder pchildwset_apos = new StringBuilder();
            //pchildwset_apos.append(pchildwset);
            //pchildwset_apos.append(" ");
            //pchildwset_apos.append(apos);
            String pchildwset_apos = pchildwset + " "+apos;
            features[index++] = pchildwset_apos;
            //StringBuilder pchildwset_adeprel = new StringBuilder();
            //pchildwset_adeprel.append(pchildwset);
            //pchildwset_adeprel.append(" ");
            //pchildwset_adeprel.append(adeprel);
            String pchildwset_adeprel = pchildwset +" "+adeprel;
            features[index++] = pchildwset_adeprel;
            //StringBuilder pchildwset_deprelpath = new StringBuilder();
            //pchildwset_deprelpath.append(pchildwset);
            //pchildwset_deprelpath.append(" ");
            //pchildwset_deprelpath.append(deprelpath);
            String pchildwset_deprelpath = pchildwset +" "+deprelpath;
            features[index++] = pchildwset_deprelpath;
            //StringBuilder pchildwset_pospath = new StringBuilder();
            //pchildwset_pospath.append(pchildwset);
            //pchildwset_pospath.append(" ");
            //pchildwset_pospath.append(pospath);
            String pchildwset_pospath = pchildwset +" "+pospath;
            features[index++] = pchildwset_pospath;
            //StringBuilder pchildwset_position = new StringBuilder();
            //pchildwset_position.append(pchildwset);
            //pchildwset_position.append(" ");
            //pchildwset_position.append(position);
            String pchildwset_position = pchildwset +" "+ position;
            features[index++] = pchildwset_position;
            //StringBuilder pchildwset_leftw = new StringBuilder();
            //pchildwset_leftw.append(pchildwset);
            //pchildwset_leftw.append(" ");
            //pchildwset_leftw.append(leftw);
            String pchildwset_leftw = pchildwset +" "+leftw;
            features[index++] = pchildwset_leftw;
            //StringBuilder pchildwset_leftpos = new StringBuilder();
            //pchildwset_leftpos.append(pchildwset);
            //pchildwset_leftpos.append(" ");
            //pchildwset_leftpos.append(leftpos);
            String pchildwset_leftpos = pchildwset +" "+leftpos;
            features[index++] = pchildwset_leftpos;
            //StringBuilder pchildwset_rightw = new StringBuilder();
            //pchildwset_rightw.append(pchildwset);
            //pchildwset_rightw.append(" ");
            //pchildwset_rightw.append(rightw);
            String pchildwset_rightw = pchildwset +" "+rightw;
            features[index++] = pchildwset_rightw;
            //StringBuilder pchildwset_rightpos = new StringBuilder();
            //pchildwset_rightpos.append(pchildwset);
            //pchildwset_rightpos.append(" ");
            //pchildwset_rightpos.append(rightpos);
            String pchildwset_rightpos = pchildwset + " "+ rightpos;
            features[index++] = pchildwset_rightpos;
            //StringBuilder pchildwset_leftsiblingw = new StringBuilder();
            //pchildwset_leftsiblingw.append(pchildwset);
            //pchildwset_leftsiblingw.append(" ");
            //pchildwset_leftsiblingw.append(leftsiblingw);
            String pchildwset_leftsiblingw = pchildwset +" "+leftsiblingw;
            features[index++] = pchildwset_leftsiblingw;
            //StringBuilder pchildwset_leftsiblingpos = new StringBuilder();
            //pchildwset_leftsiblingpos.append(pchildwset);
            //pchildwset_leftsiblingpos.append(" ");
            //pchildwset_leftsiblingpos.append(leftsiblingpos);
            String pchildwset_leftsiblingpos = pchildwset +" "+leftsiblingpos;
            features[index++] = pchildwset_leftsiblingpos;
            //StringBuilder pchildwset_rightsiblingw = new StringBuilder();
            //pchildwset_rightsiblingw.append(pchildwset);
            //pchildwset_rightsiblingw.append(" ");
            //pchildwset_rightsiblingw.append(rightsiblingw);
            String pchildwset_rightsiblingw = pchildwset +" " + rightsiblingw;
            features[index++] = pchildwset_rightsiblingw;
            //StringBuilder pchildwset_rightsiblingpos = new StringBuilder();
            //pchildwset_rightsiblingpos.append(pchildwset);
            //pchildwset_rightsiblingpos.append(" ");
            //pchildwset_rightsiblingpos.append(rightsiblingpos);
            String pchildwset_rightsiblingpos = pchildwset +" "+rightsiblingpos;
            features[index++] = pchildwset_rightsiblingpos;

            //////////////////////////////////////////////////////////////////////
            /////////// PREDICATE-ARGUMENT-ARGUMENT CONJOINED FEATURES //////////
            //////////////////////////////////////////////////////////////////////

            ///////////////////////////
            /// Plem + arg-arg ///////
            //////////////////////////
            /*
            long plem_aw_apos = ((plem<<20) | aw)<<10 | apos;
            features[index++] = plem_aw_apos;
            long plem_aw_adeprel = ((plem<<20) | aw) <<10 | adeprel;
            features[index++] = plem_aw_adeprel;
            String plem_aw_deprelpath = plem+" "+aw+" "+ deprelpath;
            features[index++] = plem_aw_deprelpath;
            String plem_aw_pospath = plem+" "+aw+" "+ pospath;
            features[index++] = plem_aw_pospath;
            long plem_aw_position = ((plem<<20) | aw) <<2 | position;
            features[index++] = plem_aw_position;
            long plem_aw_leftw = ((plem<<20)| aw) << 20 | leftw;
            features[index++] = plem_aw_leftw;
            long plem_aw_leftpos = ((plem<<20) | aw )<<10 | leftpos;
            features[index++] = plem_aw_leftpos;
            long plem_aw_rightw = ((plem<<20) | aw ) << 20 | rightw;
            features[index++] = plem_aw_rightw;
            long plem_aw_rightpos = ((plem<<20) | aw) <<10 | rightpos;
            features[index++] = plem_aw_rightpos;
            long plem_aw_leftsiblingw = ((plem<<20) | aw )<< 20 | leftsiblingw;
            features[index++] = plem_aw_leftsiblingw;
            long plem_aw_leftsiblingpos = ((plem <<20) | aw) <<10 | leftsiblingpos;
            features[index++] = plem_aw_leftsiblingpos;
            long plem_aw_rightsiblingw = ((plem <<20 ) | aw) << 20 | rightsiblingw;
            features[index++] = plem_aw_rightsiblingw;
            long plem_aw_rightsiblingpos = ((plem<<20) |aw ) <<10 | rightsiblingpos;
            features[index++] = plem_aw_rightsiblingpos;

            long plem_apos_adeprel = ((plem <<10 ) | apos) <<10 | adeprel;
            features[index++] = plem_apos_adeprel;
            String plem_apos_deprelpath = plem+" "+apos+" "+ deprelpath;
            features[index++] = plem_apos_deprelpath;
            String plem_apos_pospath = plem+" "+apos+" "+ pospath;
            features[index++] = plem_apos_pospath;
            long plem_apos_position = ((plem<<10) | apos)<<2 | position;
            features[index++] = plem_apos_position;
            long plem_apos_leftw = ((plem <<20) | leftw)<< 10 | apos;
            features[index++] = plem_apos_leftw;
            long plem_apos_leftpos = ((plem <<10 ) | apos ) <<10 | leftpos;
            features[index++] = plem_apos_leftpos;
            long plem_apos_rightw = ((plem<<20 ) | rightw)<< 10 | apos;
            features[index++] = plem_apos_rightw;
            long plem_apos_rightpos = ((plem <<10 ) | apos )<<10 | rightpos;
            features[index++] = plem_apos_rightpos;
            long plem_apos_leftsiblingw = ((plem<<20) | leftsiblingw ) << 10 | apos;
            features[index++] = plem_apos_leftsiblingw;
            long plem_apos_leftsiblingpos = ((plem <<10 ) | apos )<<10 | leftsiblingpos;
            features[index++] = plem_apos_leftsiblingpos;
            long plem_apos_rightsiblingw = ((plem <<20 ) | rightsiblingw)<< 10 | apos;
            features[index++] = plem_apos_rightsiblingw;
            long plem_apos_rightsiblingpos = ((plem <<10 ) | apos ) <<10 | rightsiblingpos;
            features[index++] = plem_apos_rightsiblingpos;

            String plem_adeprel_deprelpath = plem+ " "+adeprel+" "+ deprelpath;
            features[index++] = plem_adeprel_deprelpath;
            String plem_adeprel_pospath = plem+" "+adeprel+" "+ pospath;
            features[index++] = plem_adeprel_pospath;
            long plem_adeprel_position = ((plem <<10 )| adeprel )<<2 | position;
            features[index++] = plem_adeprel_position;
            long plem_adeprel_leftw = ((plem <<20 ) | leftw ) << 10 | adeprel;
            features[index++] = plem_adeprel_leftw;
            long plem_adeprel_leftpos = ((plem<<10 ) | adeprel ) <<10 | leftpos;
            features[index++] = plem_adeprel_leftpos;
            long plem_adeprel_rightw = ((plem <<20) | rightw ) << 10 | adeprel;
            features[index++] = plem_adeprel_rightw;
            long plem_adeprel_rightpos = ((plem <<10 )| adeprel ) <<10 | rightpos;
            features[index++] = plem_adeprel_rightpos;
            long plem_adeprel_leftsiblingw = ((plem<<20 )| leftsiblingw ) << 10 | adeprel;
            features[index++] = plem_adeprel_leftsiblingw;
            long plem_adeprel_leftsiblingpos = ((plem << 10 ) | adeprel ) <<10 | leftsiblingpos;
            features[index++] = plem_adeprel_leftsiblingpos;
            long plem_adeprel_rightsiblingw = ((plem <<20 ) |rightsiblingw ) << 10 | adeprel;
            features[index++] = plem_adeprel_rightsiblingw;
            long plem_adeprel_rightsiblingpos = ((plem <<10 ) | adeprel) <<10 | rightsiblingpos;
            features[index++] = plem_adeprel_rightsiblingpos;


            String plem_deprelpath_pospath = plem +" "+deprelpath+" "+ pospath;
            features[index++] = plem_deprelpath_pospath;
            String plem_deprelpath_position = plem +" "+deprelpath + " " +position;
            features[index++] = plem_deprelpath_position;
            String plem_deprelpath_leftw = plem +" "+leftw +" "+ deprelpath;
            features[index++] = plem_deprelpath_leftw;
            String plem_deprelpath_leftpos = plem +" "+deprelpath + " " +leftpos;
            features[index++] = plem_deprelpath_leftpos;
            String plem_deprelpath_rightw = plem +" "+rightw + " "+deprelpath;
            features[index++] = plem_deprelpath_rightw;
            String plem_deprelpath_rightpos = plem +" "+deprelpath +" " +rightpos;
            features[index++] = plem_deprelpath_rightpos;
            String plem_deprelpath_leftsiblingw = plem +" "+leftsiblingw+" "+deprelpath;
            features[index++] = plem_deprelpath_leftsiblingw;
            String plem_deprelpath_leftsiblingpos = plem +" "+deprelpath +" "+ leftsiblingpos;
            features[index++] = plem_deprelpath_leftsiblingpos;
            String plem_deprelpath_rightsiblingw = plem +" "+rightsiblingw+" "+deprelpath;
            features[index++] = plem_deprelpath_rightsiblingw;
            String plem_deprelpath_rightsiblingpos = plem +" "+deprelpath+" "+rightsiblingpos;
            features[index++] = plem_deprelpath_rightsiblingpos;


            String plem_pospath_position = plem +" "+pospath + " " +position;
            features[index++] = plem_pospath_position;
            String plem_pospath_leftw = plem +" "+leftw +" "+ pospath;
            features[index++] = plem_pospath_leftw;
            String plem_pospath_leftpos = plem +" "+pospath + " " +leftpos;
            features[index++] = plem_pospath_leftpos;
            String plem_pospath_rightw = plem +" "+rightw + " "+pospath;
            features[index++] = plem_pospath_rightw;
            String plem_pospath_rightpos = plem +" "+pospath +" " +rightpos;
            features[index++] = plem_pospath_rightpos;
            String plem_pospath_leftsiblingw = plem +" "+leftsiblingw+" "+pospath;
            features[index++] = plem_pospath_leftsiblingw;
            String plem_pospath_leftsiblingpos =plem +" "+ pospath +" "+ leftsiblingpos;
            features[index++] = plem_pospath_leftsiblingpos;
            String plem_pospath_rightsiblingw = plem +" "+rightsiblingw+" "+pospath;
            features[index++] = plem_pospath_rightsiblingw;
            String plem_pospath_rightsiblingpos = plem +" "+pospath+" "+rightsiblingpos;
            features[index++] = plem_pospath_rightsiblingpos;

            long plem_position_leftw = ((plem<<20 ) | leftw )<< 2 | position;
            features[index++] = plem_position_leftw;
            long plem_position_leftpos = ((plem <<10 ) | leftpos)<<2 | position;
            features[index++] = plem_position_leftpos;
            long plem_position_rightw = ((plem <<20 ) | rightw ) << 2 | position;
            features[index++] = plem_position_rightw;
            long plem_position_rightpos = ((plem <<10 ) | rightpos )<<2 | position;
            features[index++] = plem_position_rightpos;
            long plem_position_leftsiblingw = ((plem<<20 ) | leftsiblingw ) << 2 | position;
            features[index++] = plem_position_leftsiblingw;
            long plem_position_leftsiblingpos = ((plem<<10 )| leftsiblingpos ) <<2 | position;
            features[index++] = plem_position_leftsiblingpos;
            long plem_position_rightsiblingw = ((plem<<20) | rightsiblingw ) << 2 | position;
            features[index++] = plem_position_rightsiblingw;
            long plem_position_rightsiblingpos = ((plem << 10 ) | rightsiblingpos ) <<2 | position;
            features[index++] = plem_position_rightsiblingpos;

            long plem_leftw_leftpos = ((plem<<20) | leftw ) <<10 | leftpos;
            features[index++] = plem_leftw_leftpos;
            long plem_leftw_rightw = ((plem <<20) | leftw ) << 20 | rightw;
            features[index++] = plem_leftw_rightw;
            long plem_leftw_rightpos = ((plem <<20) | leftw ) <<10 | rightpos;
            features[index++] = plem_leftw_rightpos;
            long plem_leftw_leftsiblingw = ((plem <<20 ) | leftw ) << 20 | leftsiblingw;
            features[index++] = plem_leftw_leftsiblingw;
            long plem_leftw_leftsiblingpos = ((plem<< 20 ) | leftw ) <<10 | leftsiblingpos;
            features[index++] = plem_leftw_leftsiblingpos;
            long plem_leftw_rightsiblingw = ((plem <<20 ) | leftw ) << 20 | rightsiblingw;
            features[index++] = plem_leftw_rightsiblingw;
            long plem_leftw_rightsiblingpos = ((plem <<20 ) | leftw )<<10 | rightsiblingpos;
            features[index++] = plem_leftw_rightsiblingpos;

            long plem_leftpos_rightw = ((plem <<20) | rightw ) << 10 | leftpos;
            features[index++] = plem_leftpos_rightw;
            long plem_leftpos_rightpos = ((plem << 10 ) | leftpos ) <<10 | rightpos;
            features[index++] = plem_leftpos_rightpos;
            long plem_leftpos_leftsiblingw = ((plem << 20) | leftsiblingw ) << 10 | leftpos;
            features[index++] = plem_leftpos_leftsiblingw;
            long plem_leftpos_leftsiblingpos = ((plem << 10) | leftpos ) <<10 | leftsiblingpos;
            features[index++] = plem_leftpos_leftsiblingpos;
            long plem_leftpos_rightsiblingw = ((plem << 20)| rightsiblingw ) << 10 | leftpos;
            features[index++] = plem_leftpos_rightsiblingw;
            long plem_leftpos_rightsiblingpos = ((plem << 10) | leftpos )<<10 | rightsiblingpos;
            features[index++] = plem_leftpos_rightsiblingpos;

            long plem_rightw_rightpos = ((plem <<20) | rightw ) <<10 | rightpos;
            features[index++] = plem_rightw_rightpos;
            long plem_rightw_leftsiblingw = ((plem <<20) | rightw ) << 20 | leftsiblingw;
            features[index++] = plem_rightw_leftsiblingw;
            long plem_rightw_leftsiblingpos = ((plem<< 20)| rightw ) <<10 | leftsiblingpos;
            features[index++] = plem_rightw_leftsiblingpos;
            long plem_rightw_rightsiblingw = (( plem<<20 ) | rightw ) << 20 | rightsiblingw;
            features[index++] = plem_rightw_rightsiblingw;
            long plem_rightw_rightsiblingpos = ((plem << 20 ) | rightw ) <<10 | rightsiblingpos;
            features[index++] = plem_rightw_rightsiblingpos;


            long plem_rightpos_leftsiblingw = (( plem << 20 ) | leftsiblingw ) << 10 | rightpos;
            features[index++] = plem_rightpos_leftsiblingw;
            long plem_rightpos_leftsiblingpos = (( plem << 10 ) | rightpos ) <<10 | leftsiblingpos;
            features[index++] = plem_rightpos_leftsiblingpos;
            long plem_rightpos_rightsiblingw = (( plem <<20 ) | rightsiblingw ) << 10 | rightpos;
            features[index++] = plem_rightpos_rightsiblingw;
            long plem_rightpos_rightsiblingpos = (( plem << 10) | rightpos) <<10 | rightsiblingpos;
            features[index++] = plem_rightpos_rightsiblingpos;


            long plem_leftsiblingw_leftsiblingpos = (( plem << 20) | leftsiblingw ) <<10 | leftsiblingpos;
            features[index++] = plem_leftsiblingw_leftsiblingpos;
            long plem_leftsiblingw_rightsiblingw = (( plem << 20) | leftsiblingw ) << 20 | rightsiblingw;
            features[index++] = plem_leftsiblingw_rightsiblingw;
            long plem_leftsiblingw_rightsiblingpos = ((plem << 20) | leftsiblingw ) <<10 | rightsiblingpos;
            features[index++] = plem_leftsiblingw_rightsiblingpos;

            long plem_leftsiblingpos_rightsiblingw = ((plem << 20) | rightsiblingw ) << 10 | leftsiblingpos;
            features[index++] = plem_leftsiblingpos_rightsiblingw;
            long plem_leftsiblingpos_rightsiblingpos = ((plem << 10) | rightsiblingpos ) <<10 | leftsiblingpos;
            features[index++] = plem_leftsiblingpos_rightsiblingpos;

            long plem_rightSiblingw_rightSiblingpos = (( plem << 20) | rightsiblingw ) <<10 | rightsiblingpos;
            features[index++] = plem_rightSiblingw_rightSiblingpos;

            ///////////////////////////////////////////////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////
            //some miscellaneous tri-gram features

            int ppos_apos_adeprel = (((ppos<<10) | apos )<< 10) | adeprel;
            features[index++] = ppos_apos_adeprel;
            int pdeprel_apos_adeprel = (((pdeprel<<10) | apos )<< 10) | adeprel;
            features[index++] = pdeprel_apos_adeprel;

            String pchilddepset_apos_adeprel = pchilddepset +" "+apos+" "+adeprel;
            features[index++] = pchilddepset_apos_adeprel;

            String pchildposset_apos_adeprel = pchildposset +" "+apos +" "+adeprel;
            features[index++] = pchildposset_apos_adeprel;

            String pchildwset_apos_adeprel = pchildwset +" "+apos +" "+adeprel;
            features[index++] = pchildwset_apos_adeprel;

            String pchildwset_aw_adeprel = pchildwset +" "+aw +" "+ adeprel;
            features[index++] = pchildwset_aw_adeprel;
            */
        }


        //build feature vector for predicate disambiguation module
        if (state.equals("PD")) {
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
        }

        return features;
    }

    //TODO dependency subcat frames should contain core dep labels (not all of them)
    // todo move this as a private (non-static) member to Predicate class (i.e. member depSubCat; private function: getDepSubCat)
    private static String getDepSubCat(int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads,
                                      int[] sentenceDepLabels) {
        //StringBuilder subCat = new StringBuilder();
        String subCat = "";
        ArrayList<Integer> subCatElements= new ArrayList<Integer>();
        if (sentenceReverseDepHeads[pIdx] != null) {
            //predicate has >1 children
            for (int child : sentenceReverseDepHeads[pIdx])
                subCatElements.add(sentenceDepLabels[child]);
        }

        for (int child: subCatElements) {
            subCat += child+"\t";
        }
        return subCat.trim();
    }

    private static String getChildSet(int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads,
                                     int[] collection) {
        String childSet="";
        TreeSet<Integer> children= new TreeSet<Integer>();
        if (sentenceReverseDepHeads[pIdx] != null) {
            for (int child : sentenceReverseDepHeads[pIdx])
                children.add(collection[child]);
        }
        for (int child: children) {
            childSet += child +"\t";
        }
        return childSet.trim();
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
        if (sentenceReverseDepHeads[pIdx]!= null) {
            argSiblings = sentenceReverseDepHeads[pIdx];
        }

        if (argSiblings.lower(aIdx) != null)
            return argSiblings.lower(aIdx);
        return -1;
    }

    private static int getRightSiblingIndex(int aIdx, int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        TreeSet<Integer> argSiblings = new TreeSet<Integer>();
        if (sentenceReverseDepHeads[pIdx]!= null)
            argSiblings = sentenceReverseDepHeads[pIdx];

        if (argSiblings.higher(aIdx) != null)
            return argSiblings.higher(aIdx);
        return -1;
    }

}
