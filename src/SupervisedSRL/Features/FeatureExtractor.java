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

    public static String[] extractFeatures (Predicate p, int aIdx, Sentence sentence ,String state,int length)
    {
        String[] features= new String[length];
        String[] sentenceDepLabels= sentence.getDepLabels();
        int[] sentenceDepHeads= sentence.getDepHeads();
        String[] sentenceFeats= sentence.getFeats();
        String[] sentenceWords= sentence.getWords();
        String[] sentencePOSTags= sentence.getPosTags();
        String[] sentenceLemmas= sentence.getLemmas();
        HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads = sentence.getReverseDepHeads();

        //predicate features
        int pIdx= p.getIndex();
        String pw= sentenceWords[pIdx];
        String ppos= sentencePOSTags[pIdx];
        String plem= sentenceLemmas[pIdx];
        String pdeprel= sentenceDepLabels[pIdx];
        String pfeats = sentenceFeats[pIdx];
        String psense= p.getLabel();
        String pprw= sentenceWords[sentenceDepHeads[pIdx]];
        String pprpos= sentencePOSTags[sentenceDepHeads[pIdx]];
        String pprfeats= sentenceFeats[sentenceDepHeads[pIdx]];
        String pdepsubcat= getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels);
        String pchilddepset= getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels);
        String pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags);
        String pchildwset= getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords);

        String voice= sentence.getVoice(pIdx);

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
            features[index++] = "pw:" + pw;
            features[index++] = "ppos:" + ppos;
            features[index++] = "plem:" + plem;
            features[index++] = "pdeprel:" + pdeprel;
            features[index++] = "psense:" + psense;
            // todo features should be separated
            features[index++] = "pfeats:" + pfeats;
            features[index++] = "pprw:" + pprw;
            features[index++] = "pprpos:" + pprpos;
            features[index++] = "pprfeats:" + pprfeats;
            features[index++] = "pdepsubcat:" + pdepsubcat;
            features[index++] = "pchilddepset:" + pchilddepset;
            features[index++] = "pchildposset:" + pchildposset;
            features[index++] = "pchildwset:" + pchildwset;

            features[index++] = "aw:" + aw;
            features[index++] = "apos:" + apos;
            features[index++] = "afeat:" + afeat;
            features[index++] = "adeprel:" + adeprel;
            features[index++] = "deprelpath:" + deprelpath;
            features[index++] = "pospath:" + pospath;
            features[index++] = "position:" + position;
            features[index++] = "leftw:" + leftw;
            features[index++] = "leftpos:" + leftpos;
            features[index++] = "leftfeats:" + leftfeats;
            features[index++] = "rightw:" + rightw;
            features[index++] = "rightpos:" + rightpos;
            features[index++] = "rightfeats:" + rightfeats;
            features[index++] = "leftsiblingw:" + leftsiblingw;
            features[index++] = "leftsiblingpos:" + leftsiblingpos;
            features[index++] = "leftsiblingfeats:" + leftsiblingfeats;
            features[index++] = "rightsiblingw:" + rightsiblingw;
            features[index++] = "rightsiblingpos:" + rightsiblingpos;
            features[index++] = "rightsiblingfeats:" + rightsiblingfeats;

            //predicate-argument conjoined features
            features[index++] = "pw_aw:" + pw + "_" + aw;
            features[index++] = "pw_apos:" + pw + "_" + apos;
            features[index++] = "pw_afeat:" + pw + "_" + afeat;
            features[index++] = "pw_adeprel:" + pw + "_" + adeprel;
            features[index++] = "pw_deprelpath:" + pw + "_" + deprelpath;
            features[index++] = "pw_pospath:" + pw + "_" + pospath;
            features[index++] = "pw_position:" + pw + "_" + position;
            features[index++] = "pw_leftw:" + pw + "_" + leftw;
            features[index++] = "pw_leftpos:" + pw + "_" + leftpos;
            features[index++] = "pw_leftfeats:" + pw + "_" + leftfeats;
            features[index++] = "pw_rightw:" + pw + "_" + rightw;
            features[index++] = "pw_rightpos:" + pw + "_" + rightpos;
            features[index++] = "pw_rightfeats:" + pw + "_" + rightfeats;
            features[index++] = "pw_leftsiblingw:" + pw + "_" + leftsiblingw;
            features[index++] = "pw_leftsiblingpos:" + pw + "_" + leftsiblingpos;
            features[index++] = "pw_leftsiblingfeats:" + pw + "_" + leftsiblingfeats;
            features[index++] = "pw_rightsiblingw:" + pw + "_" + rightsiblingw;
            features[index++] = "pw_rightsiblingpos:" + pw + "_" + rightsiblingpos;
            features[index++] = "pw_rightsiblingfeats:" + pw + "_" + rightsiblingfeats;

            features[index++] = "ppos_aw:" + ppos + "_" + aw;
            features[index++] = "ppos_apos:" + ppos + "_" + apos;
            features[index++] = "ppos_afeat:" + ppos + "_" + afeat;
            features[index++] = "ppos_adeprel:" + ppos + "_" + adeprel;
            features[index++] = "ppos_deprelpath:" + ppos + "_" + deprelpath;
            features[index++] = "ppos_pospath:" + ppos + "_" + pospath;
            features[index++] = "ppos_position:" + ppos + "_" + position;
            features[index++] = "ppos_leftw:" + ppos + "_" + leftw;
            features[index++] = "ppos_leftpos:" + ppos + "_" + leftpos;
            features[index++] = "ppos_leftfeats:" + ppos + "_" + leftfeats;
            features[index++] = "ppos_rightw:" + ppos + "_" + rightw;
            features[index++] = "ppos_rightpos:" + ppos + "_" + rightpos;
            features[index++] = "ppos_rightfeats:" + ppos + "_" + rightfeats;
            features[index++] = "ppos_leftsiblingw:" + ppos + "_" + leftsiblingw;
            features[index++] = "ppos_leftsiblingpos:" + ppos + "_" + leftsiblingpos;
            features[index++] = "ppos_leftsiblingfeats:" + ppos + "_" + leftsiblingfeats;
            features[index++] = "ppos_rightsiblingw:" + ppos + "_" + rightsiblingw;
            features[index++] = "ppos_rightsiblingpos:" + ppos + "_" + rightsiblingpos;
            features[index++] = "ppos_rightsiblingfeats:" + ppos + "_" + rightsiblingfeats;


            features[index++] = "pdeprel_aw:" + pdeprel + "_" + aw;
            features[index++] = "pdeprel_apos:" + pdeprel + "_" + apos;
            features[index++] = "pdeprel_afeat:" + pdeprel + "_" + afeat;
            features[index++] = "pdeprel_adeprel:" + pdeprel + "_" + adeprel;
            features[index++] = "pdeprel_deprelpath:" + pdeprel + "_" + deprelpath;
            features[index++] = "pdeprel_pospath:" + pdeprel + "_" + pospath;
            features[index++] = "pdeprel_position:" + pdeprel + "_" + position;
            features[index++] = "pdeprel_leftw:" + pdeprel + "_" + leftw;
            features[index++] = "pdeprel_leftpos:" + pdeprel + "_" + leftpos;
            features[index++] = "pdeprel_leftfeats:" + pdeprel + "_" + leftfeats;
            features[index++] = "pdeprel_rightw:" + pdeprel + "_" + rightw;
            features[index++] = "pdeprel_rightpos:" + pdeprel + "_" + rightpos;
            features[index++] = "pdeprel_rightfeats:" + pdeprel + "_" + rightfeats;
            features[index++] = "pdeprel_leftsiblingw:" + pdeprel + "_" + leftsiblingw;
            features[index++] = "pdeprel_leftsiblingpos:" + pdeprel + "_" + leftsiblingpos;
            features[index++] = "pdeprel_leftsiblingfeats:" + pdeprel + "_" + leftsiblingfeats;
            features[index++] = "pdeprel_rightsiblingw:" + pdeprel + "_" + rightsiblingw;
            features[index++] = "pdeprel_rightsiblingpos:" + pdeprel + "_" + rightsiblingpos;
            features[index++] = "pdeprel_rightsiblingfeats:" + pdeprel + "_" + rightsiblingfeats;

            features[index++] = "pdepsubcat_aw:" + pdepsubcat + "_" + aw;
            features[index++] = "pdepsubcat_apos:" + pdepsubcat + "_" + apos;
            features[index++] = "pdepsubcat_adeprel:" + pdepsubcat + "_" + adeprel;
            features[index++] = "pdepsubcat_position:" + pdepsubcat + "_" + position;
        }

        //build feature vector for predicate disambiguation module
        if (state.equals("PD"))
        {
            int index=0;
            features[index++]= "pw:"+pw;
            features[index++]= "ppos:"+ppos;
            features[index++]= "pdeprel:"+pdeprel;
            features[index++]= "pfeats:"+pfeats;
            features[index++]= "pprw:"+pprw;
            features[index++]= "pprpos:"+pprpos;
            features[index++]= "pprfeats:"+pprfeats;
            features[index++]= "pchilddepset:"+pchilddepset;
            features[index++]= "pchildposset:"+pchildposset;
            features[index++]= "pchildwset:"+pchildwset;
        }

        return features;
    }

    //TODO dependency subcat frames should contain core dep labels (not all of them)
    public static String getDepSubCat (int pIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads,
                                     String[] sentenceDepLabels)
    {
        String subCat="";
        if (sentenceReverseDepHeads.containsKey(pIdx) && sentenceReverseDepHeads.get(pIdx).size()>0)
        {
            for (int child: sentenceReverseDepHeads.get(pIdx))
                subCat+= sentenceDepLabels[child]+"\t";
        }
        return subCat.trim().replaceAll("\t","_");
    }

    public static String getChildSet (int pIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads,
                                       String[] collection)
    {
        String subCat="";
        if (sentenceReverseDepHeads.containsKey(pIdx) && sentenceReverseDepHeads.get(pIdx).size()>0)
        {
            for (int child: sentenceReverseDepHeads.get(pIdx))
                subCat+= collection[child]+"\t";
        }
        return subCat.trim().replaceAll("\t","|");
    }

    public static int getLeftMostDependentIndex (int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads)
    {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size()>0)
            return sentenceReverseDepHeads.get(aIdx).last();
        return -1;
    }

    public static int getRightMostDependentIndex (int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads)
    {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size()>0)
            return sentenceReverseDepHeads.get(aIdx).first();
        return -1;
    }

    public static int getLeftSiblingIndex (int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads)
    {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size()>0
                && sentenceReverseDepHeads.get(aIdx).higher(aIdx)!=null)
            return sentenceReverseDepHeads.get(aIdx).higher(aIdx);
        return -1;
    }

    public static int getRightSiblingIndex (int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads)
    {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size()>0
                && sentenceReverseDepHeads.get(aIdx).lower(aIdx)!= null)
            return sentenceReverseDepHeads.get(aIdx).lower(aIdx);
        return -1;
    }


}
