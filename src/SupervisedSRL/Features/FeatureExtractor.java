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

import java.util.HashMap;
import java.util.TreeSet;
import java.util.Set;

public class FeatureExtractor {

    // todo Object[]
    public static Object[] extractFeatures(int pIdx, String pSense, int aIdx, Sentence sentence, String state, int length,
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
        boolean isPposNominal = isNominal (ppos, indexMap);
        int plem = sentenceLemmas[pIdx];
        int pdeprel = sentenceDepLabels[pIdx];
        int pprw = sentenceWords[sentenceDepHeads[pIdx]];
        int pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
        String pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels);
        String pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels);
        String pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags);
        String pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords);

        ////////////////////// AI-AC ////////////////////////////////
        if (state.equals("AI") || state.equals("AC")) {
            int leftMostDependentIndex = getLeftMostDependentIndex(aIdx, sentenceReverseDepHeads);
            int rightMostDependentIndex = getRightMostDependentIndex(aIdx, sentenceReverseDepHeads);
            int lefSiblingIndex = getLeftSiblingIndex(aIdx, sentenceReverseDepHeads);
            int rightSiblingIndex = getRightSiblingIndex(aIdx, sentenceReverseDepHeads);

            //argument features
            int aw = sentenceWords[aIdx];
            int apos = sentencePOSTags[aIdx];
            int adeprel = sentenceDepLabels[aIdx];

            //predicate-argument features
            String deprelpath = StringUtils.convertPathArrayIntoString(sentence.getDepPath(pIdx, aIdx));
            String pospath = StringUtils.convertPathArrayIntoString(sentence.getPOSPath(pIdx, aIdx));
            int position = (pIdx < aIdx) ? 1 : 0;
            int leftw = (leftMostDependentIndex != -1) ? sentenceWords[leftMostDependentIndex] : indexMap.getNullIdx();
            int leftpos = (leftMostDependentIndex != -1) ? sentencePOSTags[leftMostDependentIndex] : indexMap.getNullIdx();
            int rightw = (rightMostDependentIndex != -1) ? sentenceWords[rightMostDependentIndex] : indexMap.getNullIdx();
            int rightpos = (rightMostDependentIndex != -1) ? sentencePOSTags[rightMostDependentIndex] : indexMap.getNullIdx();
            int rightsiblingw = (rightSiblingIndex != -1) ? sentenceWords[rightSiblingIndex] : indexMap.getNullIdx();
            int rightsiblingpos = (rightSiblingIndex != -1) ? sentencePOSTags[rightSiblingIndex] : indexMap.getNullIdx();
            int leftsiblingw = (lefSiblingIndex != -1) ? sentenceWords[lefSiblingIndex] : indexMap.getNullIdx();
            int leftsiblingpos = (lefSiblingIndex != -1) ? sentencePOSTags[lefSiblingIndex] : indexMap.getNullIdx();


            if (state.equals("AI")) {
                //build feature vector for argument identification module
                int index = 0;
                if (isPposNominal==true) {
                    //nominal predicates
                    features[index++] = ppos;
                    features[index++] = plem;
                    features[index++] = pchildwset;
                    features[index++] = rightw;
                    features[index++] = rightpos;

                    int aw_ppos = (aw<<10) | ppos;
                    features[index++] = aw_ppos;
                    int ppos_apos = (ppos<<10) | apos;
                    features[index++] = ppos_apos;
                    String ppos_deprelpath = ppos+" "+deprelpath;
                    features[index++] = ppos_deprelpath;
                    String ppos_pospath= ppos +" "+pospath;
                    features[index++] = ppos_pospath;
                    int ppos_position = (ppos<<1) | position;
                    features[index++] = ppos_position;
                    int rightw_ppos = (rightw<<10) | ppos;
                    features[index++] = rightw_ppos;
                    int ppos_rightpos = (ppos<<10) | rightpos;
                    features[index++] = ppos_rightpos;

                    long aw_plem = (aw<<20) | plem;
                    features[index++] = aw_plem;
                    int plem_apos = (plem<<10) | apos;
                    features[index++] = plem_apos;
                    String plem_deprelpath = plem +" "+deprelpath;
                    features[index++] = plem_deprelpath;
                    String plem_pospath = plem +" "+pospath;
                    features[index++] = plem_pospath;
                    int plem_position = (plem<<1) | position;
                    features[index++] = plem_position;
                    long rightw_plem = (rightw<<20) | plem;
                    features[index++] = rightw_plem;
                    int plem_rightpos = (plem<<10) | rightpos;
                    features[index++] = plem_rightpos;

                    String pchildwset_aw = pchildwset +" "+ aw;
                    features[index++] = pchildwset_aw;
                    String pchildwset_apos = pchildwset + " "+apos;
                    features[index++] = pchildwset_apos;
                    String pchildwset_deprelpath = pchildwset +" "+deprelpath;
                    features[index++] = pchildwset_deprelpath;
                    String pchildwset_pospath = pchildwset +" "+pospath;
                    features[index++] = pchildwset_pospath;
                    String pchildwset_position = pchildwset +" "+ position;
                    features[index++] = pchildwset_position;
                    String pchildwset_rightw = pchildwset +" "+rightw;
                    features[index++] = pchildwset_rightw;
                    String pchildwset_rightpos = pchildwset + " "+ rightpos;
                    features[index++] = pchildwset_rightpos;

                }
                if (isPposNominal==false) {
                    //verbal predicates
                    features[index++] = pSense;
                    features[index++] = pprw;
                    features[index++] = pprpos;
                    features[index++] = adeprel;
                    features[index++] = leftsiblingw;

                    String psense_aw = pSense + " " + aw;
                    features[index++] = psense_aw;
                    String psense_apos = pSense +" "+apos;
                    features[index++] = psense_apos;
                    String psense_adeprel = pSense + " "+ adeprel;
                    features[index++] = psense_adeprel;
                    String psense_deprelpath = pSense + " "+ deprelpath;
                    features[index++] = psense_deprelpath;
                    String psense_pospath = pSense + " "+ pospath;
                    features[index++] = psense_pospath;
                    String psense_position = pSense + " " +position;
                    features[index++] = psense_position;
                    String psense_rightsiblingw = pSense +" "+rightsiblingw;
                    features[index++] = psense_rightsiblingw;

                    long aw_pprw = (aw<<20) | pprw;
                    features[index++] = aw_pprw;
                    int pprw_apos = (pprw<<10) | apos;
                    features[index++] = pprw_apos;
                    int pprw_adeprel = (pprw<<10) | adeprel;
                    features[index++] = pprw_adeprel;
                    String pprw_deprelpath = pprw +" "+deprelpath;
                    features[index++] = pprw_deprelpath;
                    String pprw_pospath = pprw +" "+ pospath;
                    features[index++] = pprw_pospath;
                    int pprw_position = (pprw<<1) | position;
                    features[index++] = pprw_position;
                    long rightsiblingw_pprw = (rightsiblingw<<20) | pprw;
                    features[index++] = rightsiblingw_pprw;

                    int aw_pprpos = (aw<<10) | pprpos;
                    features[index++] = aw_pprpos;
                    int pprpos_apos = (pprpos<<10) | apos;
                    features[index++] = pprpos_apos;
                    int pprpos_adeprel = (pprpos<<10) | adeprel;
                    features[index++] = pprpos_adeprel;
                    String pprpos_deprelpath = pprpos +" "+deprelpath;
                    features[index++] = pprpos_deprelpath;
                    String pprpos_pospath= pprpos +" "+pospath;
                    features[index++] = pprpos_pospath;
                    int pprpos_position = (pprpos<<1) | position;
                    features[index++] = pprpos_position;
                    int rightsiblingw_pprpos = (rightsiblingw<<10) | pprpos;
                    features[index++] = rightsiblingw_pprpos;
                }
                //nominal-verbal arguments
                features[index++] = aw;
                features[index++] = apos;
                features[index++] = deprelpath;
                features[index++] = pospath;
                features[index++] = position;
            }
            else if (state.equals("AC"))
            {
                //build feature vector for argument classification module
                int index = 0;
                if (isPposNominal==true) {
                    //nominal predicates
                    features[index++] = pw;
                    features[index++] = pdeprel;
                    features[index++] = pSense;
                    features[index++] = pchildposset;

                    features[index++] = aw;
                    features[index++] = apos;
                    features[index++] = position;
                    features[index++] = leftw;
                    features[index++] = rightw;
                    features[index++] = rightpos;
                    features[index++] = leftsiblingw;
                    features[index++] = leftsiblingpos;

                    long pw_aw = (pw<<20) | aw;
                    features[index++] = pw_aw;
                    int pw_apos = (pw<<10) | apos ;
                    features[index++] = pw_apos;
                    int pw_position = (pw>>8) | position;
                    features[index++] = pw_position;
                    long pw_leftw = (pw>>20) | leftw;
                    features[index++] = pw_leftw;
                    long pw_rightw = (pw>>20) | rightw;
                    features[index++] = pw_rightw;
                    int pw_rightpos = (pw>>10) | rightpos;
                    features[index++] = pw_rightpos;
                    long pw_leftsiblingw = (pw>>20) | leftsiblingw;
                    features[index++] = pw_leftsiblingw;
                    int pw_leftsiblingpos= (pw>>10) | leftsiblingpos;
                    features[index++] = pw_leftsiblingpos;

                    int aw_pdeprel = (aw<<10) | pdeprel;
                    features[index++] = aw_pdeprel;
                    int pdeprel_apos = (pdeprel<<10) | apos;
                    features[index++] = pdeprel_apos;
                    int pdeprel_position = (pdeprel<<1) | position;
                    features[index++] = pdeprel_position;
                    int leftw_pdeprel = (leftw<<10) | pdeprel;
                    features[index++] = leftw_pdeprel;
                    int rightw_pdeprel = (rightw<<10) | pdeprel;
                    features[index++] = rightw_pdeprel;
                    int leftsiblingw_pdeprel = (leftsiblingw<<10) |pdeprel;
                    features[index++] = leftsiblingw_pdeprel;
                    int pdeprel_leftsiblingpos = (pdeprel<<10) | leftsiblingpos;
                    features[index++] = pdeprel_leftsiblingpos;

                    String psense_aw = pSense + " " + aw;
                    features[index++] = psense_aw;
                    String psense_apos = pSense +" "+apos;
                    features[index++] = psense_apos;
                    String psense_position = pSense + " " +position;
                    features[index++] = psense_position;
                    String psense_leftw = pSense + " " + leftw;
                    features[index++] = psense_leftw;
                    String psense_rightw = pSense + " " + rightw;
                    features[index++] = psense_rightw;
                    String psense_rightpos = pSense + " " + rightpos;
                    features[index++] = psense_rightpos;
                    String psense_leftsiblingw = pSense +" " + leftsiblingw;
                    features[index++] = psense_leftsiblingw;
                    String psense_rightsiblingw = pSense +" "+rightsiblingw;
                    features[index++] = psense_rightsiblingw;
                    String psense_rightsiblingpos = pSense +" "+rightsiblingpos;
                    features[index++] = psense_rightsiblingpos;

                    String pchildposset_aw = pchildposset + " "+ aw;
                    features[index++] = pchildposset_aw;
                    String pchildposset_apos = pchildposset +" "+apos;
                    features[index++] = pchildposset_apos;
                    String pchildposset_position = pchildposset + " "+ position;
                    features[index++] = pchildposset_position;
                    String pchildposset_leftw = pchildposset +" "+leftw;
                    features[index++] = pchildposset_leftw;
                    String pchildposset_rightw = pchildposset + " "+rightw;
                    features[index++] = pchildposset_rightw;
                    String pchildposset_rightpos = pchildposset + " "+rightpos;
                    features[index++] = pchildposset_rightpos;
                    String pchildposset_leftsiblingw = pchildposset + " "+ leftsiblingw;
                    features[index++] = pchildposset_leftsiblingw;
                    String pchildposset_leftsiblingpos = pchildposset + " "+ leftsiblingpos;
                    features[index++] = pchildposset_leftsiblingpos;
                }
                else
                {
                    //verbal predicates
                    features[index++] = ppos;
                    features[index++] = plem;
                    features[index++] = pSense;
                    features[index++] = pprw;
                    features[index++] = pprpos;
                    features[index++] = pchilddepset;

                    features[index++] = aw;
                    features[index++] = apos;
                    features[index++] = adeprel;
                    features[index++] = deprelpath;
                    features[index++] = pospath;
                    features[index++] = position;
                    features[index++] = leftpos;
                    features[index++] = rightw;
                    features[index++] = rightpos;
                    features[index++] = leftsiblingpos;

                    int aw_ppos = (aw<<10) | ppos;
                    features[index++] = aw_ppos;
                    int ppos_apos = (ppos<<10) | apos;
                    features[index++] = ppos_apos;
                    int ppos_adeprel = (ppos<<10) | adeprel;
                    features[index++] = ppos_adeprel;
                    String ppos_deprelpath = ppos+" "+deprelpath;
                    features[index++] = ppos_deprelpath;
                    String ppos_pospath= ppos +" "+pospath;
                    features[index++] = ppos_pospath;
                    int ppos_position = (ppos<<1) | position;
                    features[index++] = ppos_position;
                    int ppos_leftpos = (ppos<<10) | leftpos;
                    features[index++] = ppos_leftpos;
                    int rightw_ppos = (rightw<<10) | ppos;
                    features[index++] = rightw_ppos;
                    int ppos_rightpos = (ppos<<10) | rightpos;
                    features[index++] = ppos_rightpos;
                    int ppos_leftsiblingpos = (ppos<<10) | leftsiblingpos;
                    features[index++] = ppos_leftsiblingpos;

                    long plem_aw = (plem<<20) | aw;
                    features[index++] = plem_aw;
                    int plem_apos = (plem<<10) | apos ;
                    features[index++] = plem_apos;
                    int plem_adeprel = (plem<<10) | adeprel;
                    features[index++] = plem_adeprel;
                    String plem_deprelpath = plem +" "+deprelpath;
                    features[index++] = plem_deprelpath;
                    String plem_pospath = plem +" "+pospath;
                    features[index++] = plem_pospath;
                    int plem_position = (plem>>8) | position;
                    features[index++] = plem_position;
                    int plem_leftpos = (plem>>10) | leftpos;
                    features[index++] = plem_leftpos;
                    long plem_rightw = (plem>>20) | rightw;
                    features[index++] = plem_rightw;
                    int plem_rightpos = (plem>>10) | rightpos;
                    features[index++] = plem_rightpos;
                    int plem_leftsiblingpos= (plem>>10) | leftsiblingpos;
                    features[index++] = plem_leftsiblingpos;

                    String psense_aw = pSense + " " + aw;
                    features[index++] = psense_aw;
                    String psense_apos = pSense +" "+apos;
                    features[index++] = psense_apos;
                    String psense_adeprel = pSense + " "+ adeprel;
                    features[index++] = psense_adeprel;
                    String psense_deprelpath = pSense + " "+ deprelpath;
                    features[index++] = psense_deprelpath;
                    String psense_pospath = pSense + " "+ pospath;
                    features[index++] = psense_pospath;
                    String psense_position = pSense + " " +position;
                    features[index++] = psense_position;
                    String psense_leftpos = pSense + " " + leftpos;
                    features[index++] = psense_leftpos;
                    String psense_rightw = pSense + " " + rightw;
                    features[index++] = psense_rightw;
                    String psense_rightpos = pSense + " " + rightpos;
                    features[index++] = psense_rightpos;
                    String psense_leftsiblingpos = pSense + " " + leftsiblingpos;
                    features[index++] = psense_leftsiblingpos;

                    long aw_pprw = (aw<<20) | pprw;
                    features[index++] = aw_pprw;
                    int pprw_apos = (pprw<<10) | apos;
                    features[index++] = pprw_apos;
                    int pprw_adeprel = (pprw<<10) | adeprel;
                    features[index++] = pprw_adeprel;
                    String pprw_deprelpath = pprw +" "+deprelpath;
                    features[index++] = pprw_deprelpath;
                    String pprw_pospath = pprw +" "+ pospath;
                    features[index++] = pprw_pospath;
                    int pprw_position = (pprw<<1) | position;
                    features[index++] = pprw_position;
                    int pprw_leftpos = (pprw<<10) | leftpos;
                    features[index++] = pprw_leftpos;
                    long rightw_pprw = (rightw<<20) | pprw;
                    features[index++] = rightw_pprw;
                    int pprw_rightpos = (pprw<<10) | rightpos;
                    features[index++] = pprw_rightpos;
                    int pprw_leftsiblingpos = (pprw<<10) | leftsiblingpos;
                    features[index++] = pprw_leftsiblingpos;

                    int aw_pprpos = (aw<<10) | pprpos;
                    features[index++] = aw_pprpos;
                    int pprpos_apos = (pprpos<<10) | apos;
                    features[index++] = pprpos_apos;
                    int pprpos_adeprel = (pprpos<<10) | adeprel;
                    features[index++] = pprpos_adeprel;
                    String pprpos_deprelpath = pprpos+" "+deprelpath;
                    features[index++] = pprpos_deprelpath;
                    String pprpos_pospath= pprpos +" "+pospath;
                    features[index++] = pprpos_pospath;
                    int pprpos_position = (pprpos<<1) | position;
                    features[index++] = pprpos_position;
                    int pprpos_leftpos = (pprpos<<10) | leftpos;
                    features[index++] = pprpos_leftpos;
                    int rightw_pprpos = (rightw<<10) | pprpos;
                    features[index++] = rightw_pprpos;
                    int pprpos_rightpos = (pprpos<<10) | rightpos;
                    features[index++] = pprpos_rightpos;
                    int pprpos_leftsiblingpos = (pprpos<<10) | leftsiblingpos;
                    features[index++] = pprpos_leftsiblingpos;

                    String pchilddepset_aw = pchilddepset +" "+aw;
                    features[index++] = pchilddepset_aw;
                    String pchilddepset_apos = pchilddepset +" "+ apos;
                    features[index++] = pchilddepset_apos;
                    String pchilddepset_adeprel = pchilddepset +" "+ adeprel;
                    features[index++] = pchilddepset_adeprel;
                    String pchilddepset_deprelpath = pchilddepset +" "+ deprelpath;
                    features[index++] = pchilddepset_deprelpath;
                    String pchilddepset_pospath = pchilddepset +" "+pospath;
                    features[index++] = pchilddepset_pospath;
                    String pchilddepset_position = pchilddepset +" "+position;
                    features[index++] = pchilddepset_position;
                    String pchilddepset_leftpos = pchilddepset +" "+leftpos;
                    features[index++] = pchilddepset_leftpos;
                    String pchilddepset_rightw = pchilddepset +" "+rightw;
                    features[index++] = pchilddepset_rightw;
                    String pchilddepset_rightpos = pchilddepset +" "+rightpos;
                    features[index++] = pchilddepset_rightpos;
                    String pchilddepset_leftsiblingpos = pchilddepset +" "+ leftsiblingpos;
                    features[index++] = pchilddepset_leftsiblingpos;

                }
            }
        }
        ////////////////////// PD ////////////////////////////////
        //build feature vector for predicate disambiguation module
        if (state.equals("PD")) {
            int index = 0;
            //features[index++] = pw;
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
        TreeSet<Integer> subCatElements= new TreeSet<Integer>();
        if (sentenceReverseDepHeads[pIdx] != null) {
            for (int child : sentenceReverseDepHeads[pIdx])
                subCatElements.add(sentenceDepLabels[child]);
        }

        for (int child: subCatElements) {
            //subCat.append(child);
            //subCat.append("\t");
            subCat += child+"\t";
        }
        return subCat.trim();
    }

    private static String getChildSet(int pIdx, TreeSet<Integer>[] sentenceReverseDepHeads,
                                     int[] collection) {
        //StringBuilder childSet = new StringBuilder();
        String childSet="";
        TreeSet<Integer> children= new TreeSet<Integer>();
        if (sentenceReverseDepHeads[pIdx] != null) {
            for (int child : sentenceReverseDepHeads[pIdx])
                children.add(collection[child]);
        }
        for (int child: children) {
            childSet += child +"\t";
            //childSet.append(child);
            //childSet.append("\t");
        }
        return childSet.trim();
    }

    private static int getLeftMostDependentIndex(int aIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads[aIdx] != null)
            return sentenceReverseDepHeads[aIdx].last();
        return -1;
    }

    private static int getRightMostDependentIndex(int aIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads[aIdx] != null)
            return sentenceReverseDepHeads[aIdx].first();
        return -1;
    }

    private static int getLeftSiblingIndex(int aIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads[aIdx] != null && sentenceReverseDepHeads[aIdx].higher(aIdx) != null)
            return sentenceReverseDepHeads[aIdx].higher(aIdx);
        return -1;
    }

    private static int getRightSiblingIndex(int aIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads[aIdx] != null && sentenceReverseDepHeads[aIdx].lower(aIdx) != null)
            return sentenceReverseDepHeads[aIdx].lower(aIdx);
        return -1;
    }

    private static boolean isNominal (int ppos, IndexMap indexMap)
    {
        String[] int2StringMap = indexMap.getInt2stringMap();
        String pos = int2StringMap[ppos];
        if (pos.startsWith("VB"))
            return false;
        else
            return true;
    }

}
