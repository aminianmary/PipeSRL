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

        //first order semantic features

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

        //argument features
        int leftMostDependentIndex = getLeftMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int rightMostDependentIndex = getRightMostDependentIndex(aIdx, sentenceReverseDepHeads);
        int lefSiblingIndex = getLeftSiblingIndex(aIdx, sentenceReverseDepHeads);
        int rightSiblingIndex = getRightSiblingIndex(aIdx, sentenceReverseDepHeads);

        String aw= sentenceWords[aIdx];
        String apos= sentencePOSTags[aIdx];
        String afeat= sentenceFeats[aIdx];
        String adeprel= sentenceDepLabels[aIdx];


        //predicate-argument features
        String deprelpath= StringUtils.convertPathArrayIntoString(sentence.getDepPath(pIdx, aIdx));
        String pospath= StringUtils.convertPathArrayIntoString(sentence.getPOSPath(pIdx, aIdx));
        String position= (pIdx < aIdx) ? "a" :"b";
        String leftw= (leftMostDependentIndex!=-1) ? sentenceWords[leftMostDependentIndex]:"";
        String leftpos= (leftMostDependentIndex!=-1) ? sentencePOSTags[leftMostDependentIndex]:"";
        String leftfeats= (leftMostDependentIndex!=-1) ? sentenceFeats[leftMostDependentIndex]:"";
        String rightw= (rightMostDependentIndex!=-1) ? sentenceWords[rightMostDependentIndex]:"";
        String rightpos= (rightMostDependentIndex!=-1) ? sentencePOSTags[rightMostDependentIndex]:"";
        String rightfeats= (rightMostDependentIndex!=-1) ? sentenceFeats[rightMostDependentIndex]:"";
        String rightsiblingw= (rightSiblingIndex!=-1 ) ? sentenceWords[rightSiblingIndex]:"";
        String rightsiblingpos= (rightSiblingIndex!=-1 ) ? sentencePOSTags[rightSiblingIndex]:"";
        String rightsiblingfeats= (rightSiblingIndex!=-1 ) ? sentenceFeats[rightSiblingIndex]:"";
        String leftsiblingw= (lefSiblingIndex!=-1 ) ? sentenceWords[lefSiblingIndex]:"";
        String leftsiblingpos= (lefSiblingIndex!=-1 ) ? sentencePOSTags[lefSiblingIndex]:"";
        String leftsiblingfeats= (lefSiblingIndex!=-1 ) ? sentenceFeats[lefSiblingIndex]:"";



        //build feature vector for argument identification module
        if (state.equals("AI"))
        {
            int index=0;
            features[index++]= "pw:"+pw;
            features[index++]= "ppos:"+ppos;
            features[index++]= "plem:"+plem;
            features[index++]= "pdeprel:"+pdeprel;
            features[index++]= "psense:"+psense;
            features[index++]= "pfeats:"+pfeats;
            features[index++]= "pprw:"+pprw;
            features[index++]= "pprpos:"+pprpos;
            features[index++]= "pprfeats:"+pprfeats;
            features[index++]= "pdepsubcat:"+pdepsubcat;
            features[index++]= "pchilddepset:"+pchilddepset;
            features[index++]= "pchildposset:"+pchildposset;
            features[index++]= "pchildwset:"+pchildwset;

            features[index++]= "aw:"+aw;
            features[index++]= "apos:"+apos;
            features[index++]= "afeat:"+afeat;
            features[index++]= "adeprel:"+adeprel;
            features[index++]= "deprelpath:"+deprelpath;
            features[index++]= "pospath:"+pospath;
            features[index++]= "position:"+position;
            features[index++]= "leftw:"+leftw;
            features[index++]= "leftpos:"+leftpos;
            features[index++]= "leftfeats:"+leftfeats;
            features[index++]= "rightw:"+rightw;
            features[index++]= "rightpos:"+rightpos;
            features[index++]= "rightfeats:"+rightfeats;
            features[index++]= "leftsiblingw:"+leftsiblingw;
            features[index++]= "leftsiblingpos:"+leftsiblingpos;
            features[index++]= "leftsiblingfeats:"+leftsiblingfeats;
            features[index++]= "rightsiblingw:"+rightsiblingw;
            features[index++]= "rightsiblingpos:"+rightsiblingpos;
            features[index++]= "rightsiblingfeats:"+rightsiblingfeats;

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
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size()>0)
            return sentenceReverseDepHeads.get(aIdx).higher(aIdx);
        return -1;
    }

    public static int getRightSiblingIndex (int aIdx, HashMap<Integer, TreeSet<Integer>> sentenceReverseDepHeads)
    {
        if (sentenceReverseDepHeads.containsKey(aIdx) && sentenceReverseDepHeads.get(aIdx).size()>0)
            return sentenceReverseDepHeads.get(aIdx).lower(aIdx);
        return -1;
    }


}
