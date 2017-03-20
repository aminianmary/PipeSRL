package SupervisedSRL.Features;
/**
 * Created by Maryam Aminian on 5/17/16.
 */

import SentenceStruct.Sentence;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Pair;
import util.StringUtils;

import java.util.ArrayList;
import java.util.HashMap;
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

    public static Object[] extractPIFeatures(int wordIdx, Sentence sentence, int length, IndexMap indexMap) throws Exception {
        Object[] features = new Object[length];
        int[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        int[] sentenceWords = sentence.getWords();
        int[] sentenceWordsFullClusters = sentence.getWordFullClusterIds();
        int[] sentenceWords4Clusters = sentence.getWord4ClusterIds();
        int[] sentenceLemmas = sentence.getLemmas();
        int[] sentenceLemmasClusters = sentence.getLemmaClusterIds();
        int[] sentencePOSTags = sentence.getPosTags();
        int[] sentenceCPOSTags = sentence.getcPosTags();
        TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();

        //word features
        int w = sentenceWords[wordIdx];
        int lem = sentenceLemmas[wordIdx];
        int wFullCluster = sentenceWordsFullClusters[wordIdx];
        int w4Clsuter = sentenceWords4Clusters[wordIdx];
        int lemCluster = sentenceLemmasClusters[wordIdx];
        int pos = sentencePOSTags[wordIdx];
        int cPos = sentenceCPOSTags[wordIdx];
        int deprel = sentenceDepLabels[wordIdx];
        int prw = sentenceWords[sentenceDepHeads[wordIdx]];
        int prlem = sentenceLemmas[sentenceDepHeads[wordIdx]];
        int prwFullCluster = sentenceWordsFullClusters[sentenceDepHeads[wordIdx]];
        int prw4Cluster = sentenceWords4Clusters[sentenceDepHeads[wordIdx]];
        int prlemaCluster = sentenceLemmasClusters[sentenceDepHeads[wordIdx]];
        int prpos = sentencePOSTags[sentenceDepHeads[wordIdx]];
        int prcpos = sentenceCPOSTags[sentenceDepHeads[wordIdx]];
        String depsubcat = getDepSubCat(wordIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String childdepset = getChildSet(wordIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String childposset = getChildSet(wordIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags, indexMap);
        String childwset = getChildSet(wordIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags, indexMap);

        ArrayList<Object> feats= new ArrayList<>();
        feats.add(w);
        feats.add(lem);
        feats.add(wFullCluster);
        feats.add(w4Clsuter);
        feats.add(lemCluster);
        feats.add(pos);
        feats.add(cPos);
        feats.add(deprel);
        feats.add(prw);
        feats.add(prwFullCluster);
        feats.add(prw4Cluster);
        feats.add(prlem);
        feats.add(prlemaCluster);
        feats.add(prpos);
        feats.add(prcpos);
        feats.add(depsubcat);
        feats.add(childdepset);
        feats.add(childposset);
        feats.add(childwset);

        int index = 0;
        features[index++] = w;
        features[index++] = lem;
        features[index++] = wFullCluster;
        features[index++] = w4Clsuter;
        features[index++] = lemCluster;
        features[index++] = pos;
        features[index++] = cPos;
        features[index++] = deprel;
        features[index++] = prw;
        features[index++] = prlem;
        features[index++] = prwFullCluster;
        features[index++] = prw4Cluster;
        features[index++] = prlemaCluster;
        features[index++] = prpos;
        features[index++] = prcpos;
        features[index++] = depsubcat;
        features[index++] = childdepset;
        features[index++] = childposset;
        features[index++] = childwset;

        for (int i=0;i<feats.size();i++){
            for (int j=0; j< feats.size(); j++){
                if (i!= j)
                    features[index++] =  feats.get(i) +" "+ feats.get(j);}
        }
        return features;
    }


    public static Object[] extractPDFeatures(int pIdx, Sentence sentence, int length, IndexMap indexMap)
            throws Exception {
        Object[] features = new Object[length];
        int[] sentenceDepLabels = sentence.getDepLabels();
        int[] sentenceDepHeads = sentence.getDepHeads();
        int[] sentenceWords = sentence.getWords();
        int[] sentenceWordsFullClusters = sentence.getWordFullClusterIds();
        int[] sentenceWords4Clusters = sentence.getWord4ClusterIds();
        int[] sentenceLemmas = sentence.getLemmas();
        int[] sentenceLemmasClusters = sentence.getLemmaClusterIds();
        int[] sentencePOSTags = sentence.getPosTags();
        int[] sentenceCPOSTags = sentence.getcPosTags();
        TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();

        //predicate features
        int pw = sentenceWords[pIdx];
        int plem = sentenceLemmas[pIdx];
        int pwFullCluster = sentenceWordsFullClusters[pIdx];
        int pw4Clsuter = sentenceWords4Clusters[pIdx];
        int plemCluster = sentenceLemmasClusters[pIdx];
        int ppos = sentencePOSTags[pIdx];
        int pcPos = sentenceCPOSTags[pIdx];
        int pdeprel = sentenceDepLabels[pIdx];
        int pprw = sentenceWords[sentenceDepHeads[pIdx]];
        int pprlem = sentenceLemmas[sentenceDepHeads[pIdx]];
        int pprwFullCluster = sentenceWordsFullClusters[sentenceDepHeads[pIdx]];
        int pprw4Cluster = sentenceWords4Clusters[sentenceDepHeads[pIdx]];
        int pprlemaCluster = sentenceLemmasClusters[sentenceDepHeads[pIdx]];
        int pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
        int pprcpos = sentenceCPOSTags[sentenceDepHeads[pIdx]];
        String pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
        String pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags, indexMap);
        String pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags, indexMap);

        ArrayList<Object> pFeats= new ArrayList<>();
        pFeats.add(pw);
        pFeats.add(plem);
        pFeats.add(pwFullCluster);
        pFeats.add(pw4Clsuter);
        pFeats.add(plemCluster);
        pFeats.add(ppos);
        pFeats.add(pcPos);
        pFeats.add(pdeprel);
        pFeats.add(pprw);
        pFeats.add(pprwFullCluster);
        pFeats.add(pprw4Cluster);
        pFeats.add(pprlem);
        pFeats.add(pprlemaCluster);
        pFeats.add(pprpos);
        pFeats.add(pprcpos);
        pFeats.add(pdepsubcat);
        pFeats.add(pchilddepset);
        pFeats.add(pchildposset);
        pFeats.add(pchildwset);

        int index = 0;
        features[index++] = pw;
        features[index++] = plem;
        features[index++] = pwFullCluster;
        features[index++] = pw4Clsuter;
        features[index++] = plemCluster;
        features[index++] = ppos;
        features[index++] = pcPos;
        features[index++] = pdeprel;
        features[index++] = pprw;
        features[index++] = pprlem;
        features[index++] = pprwFullCluster;
        features[index++] = pprw4Cluster;
        features[index++] = pprlemaCluster;
        features[index++] = pprpos;
        features[index++] = pprcpos;
        features[index++] = pdepsubcat;
        features[index++] = pchilddepset;
        features[index++] = pchildposset;
        features[index++] = pchildwset;

        for (int i=0;i<pFeats.size();i++){
            for (int j=0; j< pFeats.size(); j++){
                if (i!= j)
                    features[index++] =  pFeats.get(i) +" "+ pFeats.get(j);
            }
        }

        return features;
    }

    public static Object[] extractAIFeatures(int pIdx, int aIdx, Sentence sentence, int length,
                                             IndexMap indexMap, boolean extractGlobalFeatures, int label) throws Exception {
        Object[] features = new Object[length];
        BaseFeatureFields baseFeatureFields = new BaseFeatureFields(pIdx, aIdx, sentence, indexMap).invoke();
        Object[] predFeats = addAllPredicateFeatures(baseFeatureFields, features, 0, extractGlobalFeatures, label);
        Object[] argFeats = addAllArgumentFeatures(baseFeatureFields, (Object[]) predFeats[0], (Integer) predFeats[1], extractGlobalFeatures, label);
        Object[] bigramFeatures = addPredicateArgumentBigramFeatures(baseFeatureFields, (Object[]) argFeats[0], (Integer) argFeats[1], extractGlobalFeatures, label);

        Object[] AIFeatures = bigramFeatures;
        return (Object[]) AIFeatures[0];
    }

    public static Object[] extractACFeatures(int pIdx, int aIdx, Sentence sentence, int length,
                                             IndexMap indexMap, boolean extractGlobalFeatures, int label) throws Exception {

        Object[] features = new Object[length];
        BaseFeatureFields baseFeatureFields = new BaseFeatureFields(pIdx, aIdx, sentence, indexMap).invoke();
        Object[] predFeats = addAllPredicateFeatures(baseFeatureFields, features, 0, extractGlobalFeatures, label);
        Object[] argFeats = addAllArgumentFeatures(baseFeatureFields, (Object[]) predFeats[0], (Integer) predFeats[1], extractGlobalFeatures, label);
        Object[] bigramFeatures = addPredicateArgumentBigramFeatures(baseFeatureFields, (Object[]) argFeats[0], (Integer) argFeats[1], extractGlobalFeatures, label);

        Object[] ACFeatures = bigramFeatures;
        return (Object[]) ACFeatures[0];
    }

    public static Object[] extractJointFeatures(int pIdx, int aIdx, Sentence sentence, int length,
                                                IndexMap indexMap, boolean extractGlobalFeatures, int label) throws Exception {

        Object[] features = new Object[length];
        BaseFeatureFields baseFeatureFields = new BaseFeatureFields(pIdx, aIdx, sentence, indexMap).invoke();
        Object[] predFeats = addAllPredicateFeatures(baseFeatureFields, features, 0, extractGlobalFeatures, label);
        Object[] argFeats = addAllArgumentFeatures(baseFeatureFields, (Object[]) predFeats[0], (Integer) predFeats[1], extractGlobalFeatures, label);
        Object[] jointFeatures = addPredicateArgumentBigramFeatures(baseFeatureFields, (Object[]) argFeats[0], (Integer) argFeats[1], extractGlobalFeatures, label);
        return (Object[]) jointFeatures[0];
    }

    public static Object[] extractGlobalFeatures(int pIdx, String pLabel, Pair<Double, ArrayList<Integer>> aiCandid,
                                                 Pair<Double, ArrayList<Integer>> acCandid, String[] labelMap) {

        String seqOfCoreArgumentLabels = "";
        boolean predicateSeen = false;
        for (int i = 0; i < aiCandid.second.size(); i++) {
            int wordIdx = aiCandid.second.get(i);
            String label = labelMap[acCandid.second.get(i)];
            if (!(label.equalsIgnoreCase("A0")
                    || label.equalsIgnoreCase("A1")
                    || label.equalsIgnoreCase("A2")
                    || label.equalsIgnoreCase("A3")
                    || label.equalsIgnoreCase("A4")))
                continue;

            if (wordIdx < pIdx || predicateSeen)
                seqOfCoreArgumentLabels += label + " ";
            else if (wordIdx > pIdx) {
                seqOfCoreArgumentLabels += pLabel + " " + label + " ";
                predicateSeen = true;
            }
        }
        if (predicateSeen == false)
            seqOfCoreArgumentLabels += pLabel;
        return new Object[]{seqOfCoreArgumentLabels.trim()};
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
            int firstChild = sentenceReverseDepHeads[aIdx].first();
            // this should be on the left side.
            if (firstChild < aIdx) {
                return firstChild;
            }
        }
        return IndexMap.nullIdx;
    }

    private static int getRightMostDependentIndex(int aIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        if (sentenceReverseDepHeads[aIdx] != null) {
            int last = sentenceReverseDepHeads[aIdx].last();
            if (last > aIdx) {
                return sentenceReverseDepHeads[aIdx].last();
            }
        }
        return IndexMap.nullIdx;
    }

    private static int getLeftSiblingIndex(int aIdx, int parIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        TreeSet<Integer> argSiblings = new TreeSet<Integer>();
        if (sentenceReverseDepHeads[parIdx] != null) {
            argSiblings = sentenceReverseDepHeads[parIdx];
        }

        if (argSiblings.lower(aIdx) != null)
            return argSiblings.lower(aIdx);
        return IndexMap.nullIdx;
    }

    private static int getRightSiblingIndex(int aIdx, int parIdx, TreeSet<Integer>[] sentenceReverseDepHeads) {
        TreeSet<Integer> argSiblings = new TreeSet<Integer>();
        if (sentenceReverseDepHeads[parIdx] != null)
            argSiblings = sentenceReverseDepHeads[parIdx];

        if (argSiblings.higher(aIdx) != null)
            return argSiblings.higher(aIdx);
        return IndexMap.nullIdx;
    }


    private static Object[] addAllPredicateFeatures(BaseFeatureFields baseFeatureFields, Object[] currentFeatures,
                                                    int length, boolean extractGlobalFeatures, int label) {
        int index = length;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pw) << 6 | label : baseFeatureFields.pw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.ppos) << 6 | label : baseFeatureFields.ppos;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.plem) << 6 | label : baseFeatureFields.plem;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pdeprel) << 6 | label : baseFeatureFields.pdeprel;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pSense : baseFeatureFields.pSense;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pprw) << 6 | label : baseFeatureFields.pprw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pprpos) << 6 | label : baseFeatureFields.pprpos;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pdepsubcat : baseFeatureFields.pdepsubcat;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pchilddepset : baseFeatureFields.pchilddepset;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pchildposset : baseFeatureFields.pchildposset;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pchildwset : baseFeatureFields.pchildwset;
        //word cluster features
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pwFullCluster) << 6 | label : baseFeatureFields.pwFullCluster;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pw4cluster) << 6 | label : baseFeatureFields.pw4cluster;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.plem_cluster) << 6 | label : baseFeatureFields.plem_cluster;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pprw_fullCluster) << 6 | label : baseFeatureFields.pprw_fullCluster;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.pprw_4cluster) << 6 | label : baseFeatureFields.pprw_4cluster;

        return new Object[]{currentFeatures, index};
    }

    private static Object[] addAllArgumentFeatures(BaseFeatureFields baseFeatureFields, Object[] currentFeatures,
                                                   int length, boolean extractGlobalFeatures, int label) {
        int index = length;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.aw << 6) | label : baseFeatureFields.aw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.apos << 6) | label : baseFeatureFields.apos;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.adeprel << 6) | label : baseFeatureFields.adeprel;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.deprelpath : baseFeatureFields.deprelpath;
        currentFeatures[index++] = (extractGlobalFeatures) ? label + " " + baseFeatureFields.pospath : baseFeatureFields.pospath;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.position << 6) | label : baseFeatureFields.position;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.leftw << 6) | label : baseFeatureFields.leftw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.leftpos << 6) | label : baseFeatureFields.leftpos;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.rightw << 6) | label : baseFeatureFields.rightw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.rightpos << 6) | label : baseFeatureFields.rightpos;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.leftsiblingw << 6) | label : baseFeatureFields.leftsiblingw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.leftsiblingpos << 6) | label : baseFeatureFields.leftsiblingpos;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.rightsiblingw << 6) | label : baseFeatureFields.rightsiblingw;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.rightsiblingpos << 6) | label : baseFeatureFields.rightsiblingpos;
        //word cluster features
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.aw_fullCluster << 6) | label : baseFeatureFields.aw_fullCluster;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.aw_4cluster << 6) | label : baseFeatureFields.aw_4cluster;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.leftw_cluster << 6) | label : baseFeatureFields.leftw_cluster;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.rightw_cluster << 6) | label : baseFeatureFields.rightw_cluster;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.leftsiblingw_cluster << 6) | label : baseFeatureFields.leftsiblingw_cluster;
        currentFeatures[index++] = (extractGlobalFeatures) ? (baseFeatureFields.rightsiblingw_cluster << 6) | label : baseFeatureFields.rightsiblingw_cluster;

        return new Object[]{currentFeatures, index};
    }

    private static Object[] addPredicateArgumentBigramFeatures(BaseFeatureFields baseFeatureFields, Object[] features,
                                                               int length, boolean extractGlobalFeatures, int label) {
        int index = length;
        int pw = (extractGlobalFeatures) ? (baseFeatureFields.getPw() << 6) | label : baseFeatureFields.getPw();
        int ppos = (extractGlobalFeatures) ? (baseFeatureFields.getPpos() << 6) | label : baseFeatureFields.getPpos();
        int plem = (extractGlobalFeatures) ? (baseFeatureFields.getPlem() << 6) | label : baseFeatureFields.getPlem();
        String pSense = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getpSense() : baseFeatureFields.getpSense();
        int pdeprel = (extractGlobalFeatures) ? (baseFeatureFields.getPdeprel() << 6) | label : baseFeatureFields.getPdeprel();
        int pprw = (extractGlobalFeatures) ? (baseFeatureFields.getPprw() << 6) | label : baseFeatureFields.getPprw();
        int pprpos = (extractGlobalFeatures) ? (baseFeatureFields.getPprpos() << 6) | label : baseFeatureFields.getPprpos();
        String pdepsubcat = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getPdepsubcat() : baseFeatureFields.getPdepsubcat();
        String pchilddepset = (extractGlobalFeatures) ? label + " " + label + " " + baseFeatureFields.getPchilddepset() : baseFeatureFields.getPchilddepset();
        String pchildposset = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getPchildposset() : baseFeatureFields.getPchildposset();
        String pchildwset = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getPchildwset() : baseFeatureFields.getPchildwset();
        //cluster features
        int pw_fullCluster = (extractGlobalFeatures) ? (baseFeatureFields.getPwFullCluster() << 6) | label : baseFeatureFields.getPwFullCluster();
        int pw_4Cluster = (extractGlobalFeatures) ? (baseFeatureFields.getPw4cluster() << 6) | label : baseFeatureFields.getPw4cluster();
        int pprw_fullCluster = (extractGlobalFeatures) ? (baseFeatureFields.getPprw_fullCluster() << 6) | label : baseFeatureFields.getPprw_fullCluster();
        int pprw_4Cluster = (extractGlobalFeatures) ? (baseFeatureFields.getPprw_4cluster() << 6) | label : baseFeatureFields.getPprw_4cluster();

        int aw = baseFeatureFields.getAw();
        int apos = baseFeatureFields.getApos();
        int adeprel = baseFeatureFields.getAdeprel();
        String deprelpath = baseFeatureFields.getDeprelpath();
        String pospath = baseFeatureFields.getPospath();
        int position = baseFeatureFields.getPosition();
        int leftw = baseFeatureFields.getLeftw();
        int leftpos = baseFeatureFields.getLeftpos();
        int rightw = baseFeatureFields.getRightw();
        int rightpos = baseFeatureFields.getRightpos();
        int leftsiblingw = baseFeatureFields.getLeftsiblingw();
        int leftsiblingpos = baseFeatureFields.getLeftsiblingpos();
        int rightsiblingw = baseFeatureFields.getRightsiblingw();
        int rightsiblingpos = baseFeatureFields.getRightsiblingpos();
        //cluster features
        int aw_fullCluster = baseFeatureFields.getAw_fullCluster();
        int aw_4Cluster = baseFeatureFields.getAw_4cluster();


        // pw + argument features
        long pw_aw = (pw << 20) | aw;
        features[index++] = pw_aw;
        long pw_apos = (pw << 10) | apos;
        features[index++] = pw_apos;
        long pw_adeprel = (pw << 10) | adeprel;
        features[index++] = pw_adeprel;
        String pw_deprelpath = pw + " " + deprelpath;
        features[index++] = pw_deprelpath;
        String pw_pospath = pw + " " + pospath;
        features[index++] = pw_pospath;
        long pw_position = (pw << 2) | position;
        features[index++] = pw_position;
        long pw_leftw = (pw << 20) | leftw;
        features[index++] = pw_leftw;
        long pw_leftpos = (pw << 10) | leftpos;
        features[index++] = pw_leftpos;
        long pw_rightw = (pw << 20) | rightw;
        features[index++] = pw_rightw;
        long pw_rightpos = (pw << 10) | rightpos;
        features[index++] = pw_rightpos;
        long pw_leftsiblingw = (pw << 20) | leftsiblingw;
        features[index++] = pw_leftsiblingw;
        long pw_leftsiblingpos = (pw << 10) | leftsiblingpos;
        features[index++] = pw_leftsiblingpos;
        long pw_rightsiblingw = (pw << 20) | rightsiblingw;
        features[index++] = pw_rightsiblingw;
        long pw_rightsiblingpos = (pw << 10) | rightsiblingpos;
        features[index++] = pw_rightsiblingpos;
        long pw_fullCluster_aw_fullCluster = (pw_fullCluster << 6) | aw_fullCluster;
        features[index++] = pw_fullCluster_aw_fullCluster;
        long pw_4Cluster_aw_4Cluster = (pw_4Cluster << 6) | aw_4Cluster;
        features[index++] = pw_4Cluster_aw_4Cluster;
        long pw_fullCluster_apos = (pw_fullCluster << 10) | apos;
        features[index++] = pw_fullCluster_apos;
        long pw_4Cluster_apos = (pw_4Cluster << 10) | apos;
        features[index++] = pw_4Cluster_apos;
        long pw_fullCluster_adeprel = (pw_fullCluster << 10) | adeprel;
        features[index++] = pw_fullCluster_adeprel;
        long pw_4Cluster_adeprel = (pw_4Cluster << 10) | adeprel;
        features[index++] = pw_4Cluster_adeprel;
        long pw_fullCluster_position = (pw_fullCluster << 2) | position;
        features[index++] = pw_fullCluster_position;
        long pw_4Cluster_position = (pw_4Cluster << 2) | position;
        features[index++] = pw_4Cluster_position;

        //ppos + argument features
        long aw_ppos = (aw << 10) | ppos;
        features[index++] = aw_ppos;
        long ppos_apos = (ppos << 10) | apos;
        features[index++] = ppos_apos;
        long ppos_adeprel = (ppos << 10) | adeprel;
        features[index++] = ppos_adeprel;
        String ppos_deprelpath = ppos + " " + deprelpath;
        features[index++] = ppos_deprelpath;
        String ppos_pospath = ppos + " " + pospath;
        features[index++] = ppos_pospath;
        long ppos_position = (ppos << 2) | position;
        features[index++] = ppos_position;
        long leftw_ppos = (leftw << 10) | ppos;
        features[index++] = leftw_ppos;
        long ppos_leftpos = (ppos << 10) | leftpos;
        features[index++] = ppos_leftpos;
        long rightw_ppos = (rightw << 10) | ppos;
        features[index++] = rightw_ppos;
        long ppos_rightpos = (ppos << 10) | rightpos;
        features[index++] = ppos_rightpos;
        long leftsiblingw_ppos = (leftsiblingw << 10) | ppos;
        features[index++] = leftsiblingw_ppos;
        long ppos_leftsiblingpos = (ppos << 10) | leftsiblingpos;
        features[index++] = ppos_leftsiblingpos;
        long rightsiblingw_ppos = (rightsiblingw << 10) | ppos;
        features[index++] = rightsiblingw_ppos;
        long ppos_rightsiblingpos = (ppos << 10) | rightsiblingpos;
        features[index++] = ppos_rightsiblingpos;

        //pdeprel + argument features
        long aw_pdeprel = (aw << 10) | pdeprel;
        features[index++] = aw_pdeprel;
        long pdeprel_apos = (pdeprel << 10) | apos;
        features[index++] = pdeprel_apos;
        long pdeprel_adeprel = (pdeprel << 10) | adeprel;
        features[index++] = pdeprel_adeprel;
        String pdeprel_deprelpath = pdeprel + " " + deprelpath;
        features[index++] = pdeprel_deprelpath;
        String pdeprel_pospath = pdeprel + " " + pospath;
        features[index++] = pdeprel_pospath;
        long pdeprel_position = (pdeprel << 2) | position;
        features[index++] = pdeprel_position;
        long leftw_pdeprel = (leftw << 10) | pdeprel;
        features[index++] = leftw_pdeprel;
        long pdeprel_leftpos = (pdeprel << 10) | leftpos;
        features[index++] = pdeprel_leftpos;
        long rightw_pdeprel = (rightw << 10) | pdeprel;
        features[index++] = rightw_pdeprel;
        long pdeprel_rightpos = (pdeprel << 10) | rightpos;
        features[index++] = pdeprel_rightpos;
        long leftsiblingw_pdeprel = (leftsiblingw << 10) | pdeprel;
        features[index++] = leftsiblingw_pdeprel;
        long pdeprel_leftsiblingpos = (pdeprel << 10) | leftsiblingpos;
        features[index++] = pdeprel_leftsiblingpos;
        long rightsiblingw_pdeprel = (rightsiblingw << 10) | pdeprel;
        features[index++] = rightsiblingw_pdeprel;
        long pdeprel_rightsiblingpos = (pdeprel << 10) | rightsiblingpos;
        features[index++] = pdeprel_rightsiblingpos;


        //plem + argument features
        long aw_plem = (aw << 20) | plem;
        features[index++] = aw_plem;
        long plem_apos = (plem << 10) | apos;
        features[index++] = plem_apos;
        long plem_adeprel = (plem << 10) | adeprel;
        features[index++] = plem_adeprel;
        String plem_deprelpath = plem + " " + deprelpath;
        features[index++] = plem_deprelpath;
        String plem_pospath = plem + " " + pospath;
        features[index++] = plem_pospath;
        long plem_position = (plem << 2) | position;
        features[index++] = plem_position;
        long leftw_plem = (leftw << 20) | plem;
        features[index++] = leftw_plem;
        long plem_leftpos = (plem << 10) | leftpos;
        features[index++] = plem_leftpos;
        long rightw_plem = (rightw << 20) | plem;
        features[index++] = rightw_plem;
        long plem_rightpos = (plem << 10) | rightpos;
        features[index++] = plem_rightpos;
        long leftsiblingw_plem = (leftsiblingw << 20) | plem;
        features[index++] = leftsiblingw_plem;
        long plem_leftsiblingpos = (plem << 10) | leftsiblingpos;
        features[index++] = plem_leftsiblingpos;
        long rightsiblingw_plem = (rightsiblingw << 20) | plem;
        features[index++] = rightsiblingw_plem;
        long plem_rightsiblingpos = (plem << 10) | rightsiblingpos;
        features[index++] = plem_rightsiblingpos;

        //psense + argument features
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
        String psense_aw_fullCluster = pSense + " " + aw_fullCluster;
        features[index++] = psense_aw_fullCluster;
        String psense_aw_4Cluster = pSense + " " + aw_4Cluster;
        features[index++] = psense_aw_4Cluster;

        //pprw  + argument features
        long aw_pprw = (aw << 20) | pprw;
        features[index++] = aw_pprw;
        long pprw_apos = (pprw << 10) | apos;
        features[index++] = pprw_apos;
        long pprw_adeprel = (pprw << 10) | adeprel;
        features[index++] = pprw_adeprel;
        String pprw_deprelpath = pprw + " " + deprelpath;
        features[index++] = pprw_deprelpath;
        String pprw_pospath = pprw + " " + pospath;
        features[index++] = pprw_pospath;
        long pprw_position = (pprw << 2) | position;
        features[index++] = pprw_position;
        long leftw_pprw = (leftw << 20) | pprw;
        features[index++] = leftw_pprw;
        long pprw_leftpos = (pprw << 10) | leftpos;
        features[index++] = pprw_leftpos;
        long rightw_pprw = (rightw << 20) | pprw;
        features[index++] = rightw_pprw;
        long pprw_rightpos = (pprw << 10) | rightpos;
        features[index++] = pprw_rightpos;
        long leftsiblingw_pprw = (leftsiblingw << 20) | pprw;
        features[index++] = leftsiblingw_pprw;
        long pprw_leftsiblingpos = (pprw << 10) | leftsiblingpos;
        features[index++] = pprw_leftsiblingpos;
        long rightsiblingw_pprw = (rightsiblingw << 20) | pprw;
        features[index++] = rightsiblingw_pprw;
        long pprw_rightsiblingpos = (pprw << 10) | rightsiblingpos;
        features[index++] = pprw_rightsiblingpos;
        long aw_pprw_fullCluster = (aw << 26) | pprw_fullCluster;
        features[index++] = aw_pprw_fullCluster;
        long aw_pprw_4Cluster = (aw << 26) | pprw_4Cluster;
        features[index++] = aw_pprw_4Cluster;

        //pprpos + argument features
        long aw_pprpos = (aw << 10) | pprpos;
        features[index++] = aw_pprpos;
        long pprpos_apos = (pprpos << 10) | apos;
        features[index++] = pprpos_apos;
        long pprpos_adeprel = (pprpos << 10) | adeprel;
        features[index++] = pprpos_adeprel;
        String pprpos_deprelpath = pprpos + " " + deprelpath;
        features[index++] = pprpos_deprelpath;
        String pprpos_pospath = pprpos + " " + pospath;
        features[index++] = pprpos_pospath;
        long pprpos_position = (pprpos << 2) | position;
        features[index++] = pprpos_position;
        long leftw_pprpos = (leftw << 10) | pprpos;
        features[index++] = leftw_pprpos;
        long pprpos_leftpos = (pprpos << 10) | leftpos;
        features[index++] = pprpos_leftpos;
        long rightw_pprpos = (rightw << 10) | pprpos;
        features[index++] = rightw_pprpos;
        long pprpos_rightpos = (pprpos << 10) | rightpos;
        features[index++] = pprpos_rightpos;
        long leftsiblingw_pprpos = (leftsiblingw << 10) | pprpos;
        features[index++] = leftsiblingw_pprpos;
        long pprpos_leftsiblingpos = (pprpos << 10) | leftsiblingpos;
        features[index++] = pprpos_leftsiblingpos;
        long rightsiblingw_pprpos = (rightsiblingw << 10) | pprpos;
        features[index++] = rightsiblingw_pprpos;
        long pprpos_rightsiblingpos = (pprpos << 10) | rightsiblingpos;
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

        //pdepsubcat + argument features
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
        return new Object[]{features, index};
    }

    private static Object[] addPredicatePredicateBigramFeatures(BaseFeatureFields baseFeatureFields, Object[] features, int length) {
        int index = length;

        int pw = baseFeatureFields.getPw();
        int ppos = baseFeatureFields.getPpos();
        int plem = baseFeatureFields.getPlem();
        String pSense = baseFeatureFields.getpSense();
        int pdeprel = baseFeatureFields.getPdeprel();
        int pprw = baseFeatureFields.getPprw();
        int pprpos = baseFeatureFields.getPprpos();
        String pdepsubcat = baseFeatureFields.getPdepsubcat();
        String pchilddepset = baseFeatureFields.getPchilddepset();
        String pchildposset = baseFeatureFields.getPchildposset();
        String pchildwset = baseFeatureFields.getPchildwset();

        int pw_ppos = (pw << 10) | ppos;
        features[index++] = pw_ppos;
        long pw_plem = (pw << 20) | plem;
        features[index++] = pw_plem;
        int pw_pdeprel = (pw << 10) | pdeprel;
        features[index++] = pw_pdeprel;
        String pw_psense = pw + " " + pSense;
        features[index++] = pw_psense;
        long pw_pprw = (pw << 20) | pprw;
        features[index++] = pw_pprw;
        int pw_pprpos = (pw << 10) | pprpos;
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
        int ppos_plem = (plem << 10) | ppos;
        features[index++] = ppos_plem;
        int ppos_pdeprel = (ppos << 10) | pdeprel;
        features[index++] = ppos_pdeprel;
        String ppos_psense = ppos + " " + pSense;
        features[index++] = ppos_psense;
        int ppos_pprw = (pprw << 10) | ppos;
        features[index++] = ppos_pprw;
        int ppos_pprpos = (ppos << 10) | pprpos;
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
        int plem_pdeprel = (plem << 10) | pdeprel;
        features[index++] = plem_pdeprel;
        String plem_psense = plem + " " + pSense;
        features[index++] = plem_psense;
        long plem_pprw = (plem << 20) | pprw;
        features[index++] = plem_pprw;
        int plem_pprpos = (plem << 10) | pprpos;
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
        int pdeprel_pprw = (pprw << 10) | pdeprel;
        features[index++] = pdeprel_pprw;
        int pdeprel_pprpos = (pdeprel << 10) | pprpos;
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
        String psense_pprw = pSense + " " + pprw;
        features[index++] = psense_pprw;
        String psense_pprpos = pSense + " " + pprpos;
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
        int pprw_pprpos = (pprw << 10) | pprpos;
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
        return new Object[]{features, index};
    }

    private static Object[] addArgumentArgumentBigramFeatures(BaseFeatureFields baseFeatureFields, Object[] features, int length) {
        int index = length;
        int aw = baseFeatureFields.getAw();
        int apos = baseFeatureFields.getApos();
        int adeprel = baseFeatureFields.getAdeprel();
        String deprelpath = baseFeatureFields.getDeprelpath();
        String pospath = baseFeatureFields.getPospath();
        int position = baseFeatureFields.getPosition();
        int leftw = baseFeatureFields.getLeftw();
        int leftpos = baseFeatureFields.getLeftpos();
        int rightw = baseFeatureFields.getRightw();
        int rightpos = baseFeatureFields.getRightpos();
        int leftsiblingw = baseFeatureFields.getLeftsiblingw();
        int leftsiblingpos = baseFeatureFields.getLeftsiblingpos();
        int rightsiblingw = baseFeatureFields.getRightsiblingw();
        int rightsiblingpos = baseFeatureFields.getRightsiblingpos();

        int aw_apos = (aw << 10) | apos;
        features[index++] = aw_apos;
        int aw_adeprel = (aw << 10) | adeprel;
        features[index++] = aw_adeprel;
        String aw_deprelpath = aw + " " + deprelpath;
        features[index++] = aw_deprelpath;
        String aw_pospath = aw + " " + pospath;
        features[index++] = aw_pospath;
        int aw_position = (aw << 2) | position;
        features[index++] = aw_position;
        long aw_leftw = (aw << 20) | leftw;
        features[index++] = aw_leftw;
        int aw_leftpos = (aw << 10) | leftpos;
        features[index++] = aw_leftpos;
        long aw_rightw = (aw << 20) | rightw;
        features[index++] = aw_rightw;
        int aw_rightpos = (aw << 10) | rightpos;
        features[index++] = aw_rightpos;
        long aw_leftsiblingw = (aw << 20) | leftsiblingw;
        features[index++] = aw_leftsiblingw;
        int aw_leftsiblingpos = (aw << 10) | leftsiblingpos;
        features[index++] = aw_leftsiblingpos;
        long aw_rightsiblingw = (aw << 20) | rightsiblingw;
        features[index++] = aw_rightsiblingw;
        int aw_rightsiblingpos = (aw << 10) | rightsiblingpos;
        features[index++] = aw_rightsiblingpos;
        int apos_adeprel = (apos << 10) | adeprel;
        features[index++] = apos_adeprel;
        String apos_deprelpath = apos + " " + deprelpath;
        features[index++] = apos_deprelpath;
        String apos_pospath = apos + " " + pospath;
        features[index++] = apos_pospath;
        int apos_position = (apos << 2) | position;
        features[index++] = apos_position;
        int apos_leftw = (leftw << 10) | apos;
        features[index++] = apos_leftw;
        int apos_leftpos = (apos << 10) | leftpos;
        features[index++] = apos_leftpos;
        int apos_rightw = (rightw << 10) | apos;
        features[index++] = apos_rightw;
        int apos_rightpos = (apos << 10) | rightpos;
        features[index++] = apos_rightpos;
        int apos_leftsiblingw = (leftsiblingw << 10) | apos;
        features[index++] = apos_leftsiblingw;
        int apos_leftsiblingpos = (apos << 10) | leftsiblingpos;
        features[index++] = apos_leftsiblingpos;
        int apos_rightsiblingw = (rightsiblingw << 10) | apos;
        features[index++] = apos_rightsiblingw;
        int apos_rightsiblingpos = (apos << 10) | rightsiblingpos;
        features[index++] = apos_rightsiblingpos;
        String adeprel_deprelpath = adeprel + " " + deprelpath;
        features[index++] = adeprel_deprelpath;
        String adeprel_pospath = adeprel + " " + pospath;
        features[index++] = adeprel_pospath;
        int adeprel_position = (adeprel << 2) | position;
        features[index++] = adeprel_position;
        int adeprel_leftw = (leftw << 10) | adeprel;
        features[index++] = adeprel_leftw;
        int adeprel_leftpos = (adeprel << 10) | leftpos;
        features[index++] = adeprel_leftpos;
        int adeprel_rightw = (rightw << 10) | adeprel;
        features[index++] = adeprel_rightw;
        int adeprel_rightpos = (adeprel << 10) | rightpos;
        features[index++] = adeprel_rightpos;
        int adeprel_leftsiblingw = (leftsiblingw << 10) | adeprel;
        features[index++] = adeprel_leftsiblingw;
        int adeprel_leftsiblingpos = (adeprel << 10) | leftsiblingpos;
        features[index++] = adeprel_leftsiblingpos;
        int adeprel_rightsiblingw = (rightsiblingw << 10) | adeprel;
        features[index++] = adeprel_rightsiblingw;
        int adeprel_rightsiblingpos = (adeprel << 10) | rightsiblingpos;
        features[index++] = adeprel_rightsiblingpos;
        String deprelpath_pospath = deprelpath + " " + pospath;
        features[index++] = deprelpath_pospath;
        String deprelpath_position = deprelpath + " " + position;
        features[index++] = deprelpath_position;
        String deprelpath_leftw = leftw + " " + deprelpath;
        features[index++] = deprelpath_leftw;
        String deprelpath_leftpos = deprelpath + " " + leftpos;
        features[index++] = deprelpath_leftpos;
        String deprelpath_rightw = rightw + " " + deprelpath;
        features[index++] = deprelpath_rightw;
        String deprelpath_rightpos = deprelpath + " " + rightpos;
        features[index++] = deprelpath_rightpos;
        String deprelpath_leftsiblingw = leftsiblingw + " " + deprelpath;
        features[index++] = deprelpath_leftsiblingw;
        String deprelpath_leftsiblingpos = deprelpath + " " + leftsiblingpos;
        features[index++] = deprelpath_leftsiblingpos;
        String deprelpath_rightsiblingw = rightsiblingw + " " + deprelpath;
        features[index++] = deprelpath_rightsiblingw;
        String deprelpath_rightsiblingpos = deprelpath + " " + rightsiblingpos;
        features[index++] = deprelpath_rightsiblingpos;
        String pospath_position = pospath + " " + position;
        features[index++] = pospath_position;
        String pospath_leftw = leftw + " " + pospath;
        features[index++] = pospath_leftw;
        String pospath_leftpos = pospath + " " + leftpos;
        features[index++] = pospath_leftpos;
        String pospath_rightw = rightw + " " + pospath;
        features[index++] = pospath_rightw;
        String pospath_rightpos = pospath + " " + rightpos;
        features[index++] = pospath_rightpos;
        String pospath_leftsiblingw = leftsiblingw + " " + pospath;
        features[index++] = pospath_leftsiblingw;
        String pospath_leftsiblingpos = pospath + " " + leftsiblingpos;
        features[index++] = pospath_leftsiblingpos;
        String pospath_rightsiblingw = rightsiblingw + " " + pospath;
        features[index++] = pospath_rightsiblingw;
        String pospath_rightsiblingpos = pospath + " " + rightsiblingpos;
        features[index++] = pospath_rightsiblingpos;
        int position_leftw = (leftw << 2) | position;
        features[index++] = position_leftw;
        int position_leftpos = (leftpos << 2) | position;
        features[index++] = position_leftpos;
        int position_rightw = (rightw << 2) | position;
        features[index++] = position_rightw;
        int position_rightpos = (rightpos << 2) | position;
        features[index++] = position_rightpos;
        int position_leftsiblingw = (leftsiblingw << 2) | position;
        features[index++] = position_leftsiblingw;
        int position_leftsiblingpos = (leftsiblingpos << 2) | position;
        features[index++] = position_leftsiblingpos;
        int position_rightsiblingw = (rightsiblingw << 2) | position;
        features[index++] = position_rightsiblingw;
        int position_rightsiblingpos = (rightsiblingpos << 2) | position;
        features[index++] = position_rightsiblingpos;
        int leftw_leftpos = (leftw << 10) | leftpos;
        features[index++] = leftw_leftpos;
        long leftw_rightw = (leftw << 20) | rightw;
        features[index++] = leftw_rightw;
        int leftw_rightpos = (leftw << 10) | rightpos;
        features[index++] = leftw_rightpos;
        long leftw_leftsiblingw = (leftw << 20) | leftsiblingw;
        features[index++] = leftw_leftsiblingw;
        int leftw_leftsiblingpos = (leftw << 10) | leftsiblingpos;
        features[index++] = leftw_leftsiblingpos;
        long leftw_rightsiblingw = (leftw << 20) | rightsiblingw;
        features[index++] = leftw_rightsiblingw;
        int leftw_rightsiblingpos = (leftw << 10) | rightsiblingpos;
        features[index++] = leftw_rightsiblingpos;
        int leftpos_rightw = (rightw << 10) | leftpos;
        features[index++] = leftpos_rightw;
        int leftpos_rightpos = (leftpos << 10) | rightpos;
        features[index++] = leftpos_rightpos;
        int leftpos_leftsiblingw = (leftsiblingw << 10) | leftpos;
        features[index++] = leftpos_leftsiblingw;
        int leftpos_leftsiblingpos = (leftpos << 10) | leftsiblingpos;
        features[index++] = leftpos_leftsiblingpos;
        int leftpos_rightsiblingw = (rightsiblingw << 10) | leftpos;
        features[index++] = leftpos_rightsiblingw;
        int leftpos_rightsiblingpos = (leftpos << 10) | rightsiblingpos;
        features[index++] = leftpos_rightsiblingpos;
        int rightw_rightpos = (rightw << 10) | rightpos;
        features[index++] = rightw_rightpos;
        long rightw_leftsiblingw = (rightw << 20) | leftsiblingw;
        features[index++] = rightw_leftsiblingw;
        int rightw_leftsiblingpos = (rightw << 10) | leftsiblingpos;
        features[index++] = rightw_leftsiblingpos;
        long rightw_rightsiblingw = (rightw << 20) | rightsiblingw;
        features[index++] = rightw_rightsiblingw;
        int rightw_rightsiblingpos = (rightw << 10) | rightsiblingpos;
        features[index++] = rightw_rightsiblingpos;
        int rightpos_leftsiblingw = (leftsiblingw << 10) | rightpos;
        features[index++] = rightpos_leftsiblingw;
        int rightpos_leftsiblingpos = (rightpos << 10) | leftsiblingpos;
        features[index++] = rightpos_leftsiblingpos;
        int rightpos_rightsiblingw = (rightsiblingw << 10) | rightpos;
        features[index++] = rightpos_rightsiblingw;
        int rightpos_rightsiblingpos = (rightpos << 10) | rightsiblingpos;
        features[index++] = rightpos_rightsiblingpos;
        int leftsiblingw_leftsiblingpos = (leftsiblingw << 10) | leftsiblingpos;
        features[index++] = leftsiblingw_leftsiblingpos;
        long leftsiblingw_rightsiblingw = (leftsiblingw << 20) | rightsiblingw;
        features[index++] = leftsiblingw_rightsiblingw;
        int leftsiblingw_rightsiblingpos = (leftsiblingw << 10) | rightsiblingpos;
        features[index++] = leftsiblingw_rightsiblingpos;
        long leftsiblingpos_rightsiblingw = (rightsiblingw << 10) | leftsiblingpos;
        features[index++] = leftsiblingpos_rightsiblingw;
        int leftsiblingpos_rightsiblingpos = (rightsiblingpos << 10) | leftsiblingpos;
        features[index++] = leftsiblingpos_rightsiblingpos;
        int rightSiblingw_rightSiblingpos = (rightsiblingw << 10) | rightsiblingpos;
        features[index++] = rightSiblingw_rightSiblingpos;
        return new Object[]{features, index};
    }

    private static Object[] addBigramFeatures4AIFromNuguesSystem(BaseFeatureFields baseFeatureFields, Object[] features,
                                                                 int length, boolean extractGlobalFeatures, int label) {
        int index = length;
        int pw = (extractGlobalFeatures) ? (baseFeatureFields.getPw() << 6) | label : baseFeatureFields.getPw();
        String pSense = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getpSense() : baseFeatureFields.getpSense();
        int pdeprel = (extractGlobalFeatures) ? (baseFeatureFields.getPdeprel() << 6) | label : baseFeatureFields.getPdeprel();
        int pprw = (extractGlobalFeatures) ? (baseFeatureFields.getPprw() << 6) | label : baseFeatureFields.getPprw();
        String pchilddepset = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getPchilddepset() : baseFeatureFields.getPchilddepset();
        String pchildwset = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getPchildwset() : baseFeatureFields.getPchildwset();
        int aw = (extractGlobalFeatures) ? (baseFeatureFields.getAw() << 6) | label : baseFeatureFields.getAw();
        int apos = (extractGlobalFeatures) ? (baseFeatureFields.getApos() << 6) | label : baseFeatureFields.getApos();
        int adeprel = (extractGlobalFeatures) ? (baseFeatureFields.getAdeprel() << 6) | label : baseFeatureFields.getAdeprel();
        String deprelpath = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getDeprelpath() : baseFeatureFields.getDeprelpath();
        String pospath = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getPospath() : baseFeatureFields.getPospath();
        int position = (extractGlobalFeatures) ? (baseFeatureFields.getPosition() << 6) | label : baseFeatureFields.getPosition();
        int leftpos = (extractGlobalFeatures) ? (baseFeatureFields.getLeftpos() << 6) | label : baseFeatureFields.getLeftpos();
        int rightpos = (extractGlobalFeatures) ? (baseFeatureFields.getRightpos() << 6) | label : baseFeatureFields.getRightpos();
        int rightsiblingpos = (extractGlobalFeatures) ? (baseFeatureFields.getRightsiblingpos() << 6) | label : baseFeatureFields.getRightsiblingpos();

        long aw_position = (aw << 2) | position;
        features[index++] = aw_position;
        String psense_position = pSense + " " + position;
        features[index++] = psense_position;
        String psense_apos = pSense + " " + apos;
        features[index++] = psense_apos;
        String deprelpath_position = deprelpath + " " + position;
        features[index++] = deprelpath_position;
        long pprw_adeprel = (pprw << 10) | adeprel;
        features[index++] = pprw_adeprel;
        long pw_leftpos = (pw << 10) | leftpos;
        features[index++] = pw_leftpos;
        String psense_pchildwset = pSense + " " + pchildwset;
        features[index++] = psense_pchildwset;
        String adeprel_deprelpath = adeprel + " " + deprelpath;
        features[index++] = adeprel_deprelpath;
        String pospath_rightsiblingpos = pospath + " " + rightsiblingpos;
        features[index++] = pospath_rightsiblingpos;
        String pchilddepset_adeprel = pchilddepset + " " + adeprel;
        features[index++] = pchilddepset_adeprel;
        long apos_adeprel = (apos << 10) | adeprel;
        features[index++] = apos_adeprel;
        long leftpos_rightpos = (leftpos << 10) | rightpos;
        features[index++] = leftpos_rightpos;
        long pdeprel_adeprel = (pdeprel << 10) | adeprel;
        features[index++] = pdeprel_adeprel;
        return new Object[]{features, index};
    }

    private static Object[] addBigramFeatures4ACFromNuguesSystem(BaseFeatureFields baseFeatureFields, Object[] features,
                                                                 int length, boolean extractGlobalFeatures, int label) {
        int index = length;
        int plem = (extractGlobalFeatures) ? (baseFeatureFields.getPlem() << 6) | label : baseFeatureFields.getPlem();
        String pSense = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getpSense() : baseFeatureFields.getpSense();
        String pchilddepset = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getPchilddepset() : baseFeatureFields.getPchilddepset();
        int aw = (extractGlobalFeatures) ? (baseFeatureFields.getAw() << 6) | label : baseFeatureFields.getAw();
        int apos = (extractGlobalFeatures) ? (baseFeatureFields.getApos() << 6) | label : baseFeatureFields.getApos();
        int adeprel = (extractGlobalFeatures) ? (baseFeatureFields.getAdeprel() << 6) | label : baseFeatureFields.getAdeprel();
        String pospath = (extractGlobalFeatures) ? label + " " + baseFeatureFields.getPospath() : baseFeatureFields.getPospath();
        int position = (extractGlobalFeatures) ? (baseFeatureFields.getPosition() << 6) | label : baseFeatureFields.getPosition();
        int leftpos = (extractGlobalFeatures) ? (baseFeatureFields.getLeftpos() << 6) | label : baseFeatureFields.getLeftpos();
        int rightpos = (extractGlobalFeatures) ? (baseFeatureFields.getRightpos() << 6) | label : baseFeatureFields.getRightpos();
        int rightsiblingpos = (extractGlobalFeatures) ? (baseFeatureFields.getRightsiblingpos() << 6) | label : baseFeatureFields.getRightsiblingpos();
        int leftsiblingpos = (extractGlobalFeatures) ? (baseFeatureFields.getLeftsiblingpos() << 6) | label : baseFeatureFields.getLeftsiblingpos();

        String aw_psense = aw + " " + pSense;
        features[index++] = aw_psense;
        String psense_apos = pSense + " " + apos;
        features[index++] = psense_apos;
        String posPath_pSense = pospath + " " + pSense;
        features[index++] = posPath_pSense;
        long aw_position = (aw << 2) | position;
        features[index++] = aw_position;
        String psense_position = pSense + " " + position;
        features[index++] = psense_position;
        String pSense_rightSiblingPos = pSense + " " + rightsiblingpos;
        features[index++] = pSense_rightSiblingPos;
        long leftPos_rightSiblingPos = (leftpos << 10) | rightsiblingpos;
        features[index++] = leftPos_rightSiblingPos;
        long apos_position = (apos << 2) | position;
        features[index++] = apos_position;
        String pSense_leftSiblingPos = pSense + " " + leftsiblingpos;
        features[index++] = pSense_leftSiblingPos;
        String pSense_aDepRel = pSense + " " + adeprel;
        features[index++] = pSense_aDepRel;
        String pchilddepset_position = pchilddepset + " " + position;
        features[index++] = pchilddepset_position;
        long adeprel_rightPos = (adeprel << 10) | rightpos;
        features[index++] = adeprel_rightPos;
        long aw_apos = (aw << 20) | apos;
        features[index++] = aw_apos;
        long plemm_apos = (plem << 10) | apos;
        features[index++] = plemm_apos;
        long apos_adeprel = (apos << 10) | adeprel;
        features[index++] = apos_adeprel;

        return new Object[]{features, index};
    }

    private static Object[] addTrigramFeatures(BaseFeatureFields baseFeatureFields, Object[] features, int length) {
        int index = length;
        int ppos = baseFeatureFields.getPpos();
        int plem = baseFeatureFields.getPlem();
        int pdeprel = baseFeatureFields.getPdeprel();
        String pchilddepset = baseFeatureFields.getPchilddepset();
        String pchildposset = baseFeatureFields.getPchildposset();
        String pchildwset = baseFeatureFields.getPchildwset();
        int aw = baseFeatureFields.getAw();
        int apos = baseFeatureFields.getApos();
        int adeprel = baseFeatureFields.getAdeprel();
        String deprelpath = baseFeatureFields.getDeprelpath();
        String pospath = baseFeatureFields.getPospath();
        int position = baseFeatureFields.getPosition();
        int leftw = baseFeatureFields.getLeftw();
        int leftpos = baseFeatureFields.getLeftpos();
        int rightw = baseFeatureFields.getRightw();
        int rightpos = baseFeatureFields.getRightpos();
        int leftsiblingw = baseFeatureFields.getLeftsiblingw();
        int leftsiblingpos = baseFeatureFields.getLeftsiblingpos();
        int rightsiblingw = baseFeatureFields.getRightsiblingw();
        int rightsiblingpos = baseFeatureFields.getRightsiblingpos();

        ///////////////////////////
        /// Plem + arg-arg ///////
        //////////////////////////

        long plem_aw_apos = ((plem << 20) | aw) << 10 | apos;
        features[index++] = plem_aw_apos;
        long plem_aw_adeprel = ((plem << 20) | aw) << 10 | adeprel;
        features[index++] = plem_aw_adeprel;
        String plem_aw_deprelpath = plem + " " + aw + " " + deprelpath;
        features[index++] = plem_aw_deprelpath;
        String plem_aw_pospath = plem + " " + aw + " " + pospath;
        features[index++] = plem_aw_pospath;
        long plem_aw_position = ((plem << 20) | aw) << 2 | position;
        features[index++] = plem_aw_position;
        long plem_aw_leftw = ((plem << 20) | aw) << 20 | leftw;
        features[index++] = plem_aw_leftw;
        long plem_aw_leftpos = ((plem << 20) | aw) << 10 | leftpos;
        features[index++] = plem_aw_leftpos;
        long plem_aw_rightw = ((plem << 20) | aw) << 20 | rightw;
        features[index++] = plem_aw_rightw;
        long plem_aw_rightpos = ((plem << 20) | aw) << 10 | rightpos;
        features[index++] = plem_aw_rightpos;
        long plem_aw_leftsiblingw = ((plem << 20) | aw) << 20 | leftsiblingw;
        features[index++] = plem_aw_leftsiblingw;
        long plem_aw_leftsiblingpos = ((plem << 20) | aw) << 10 | leftsiblingpos;
        features[index++] = plem_aw_leftsiblingpos;
        long plem_aw_rightsiblingw = ((plem << 20) | aw) << 20 | rightsiblingw;
        features[index++] = plem_aw_rightsiblingw;
        long plem_aw_rightsiblingpos = ((plem << 20) | aw) << 10 | rightsiblingpos;
        features[index++] = plem_aw_rightsiblingpos;
        long plem_apos_adeprel = ((plem << 10) | apos) << 10 | adeprel;
        features[index++] = plem_apos_adeprel;
        String plem_apos_deprelpath = plem + " " + apos + " " + deprelpath;
        features[index++] = plem_apos_deprelpath;
        String plem_apos_pospath = plem + " " + apos + " " + pospath;
        features[index++] = plem_apos_pospath;
        long plem_apos_position = ((plem << 10) | apos) << 2 | position;
        features[index++] = plem_apos_position;
        long plem_apos_leftw = ((plem << 20) | leftw) << 10 | apos;
        features[index++] = plem_apos_leftw;
        long plem_apos_leftpos = ((plem << 10) | apos) << 10 | leftpos;
        features[index++] = plem_apos_leftpos;
        long plem_apos_rightw = ((plem << 20) | rightw) << 10 | apos;
        features[index++] = plem_apos_rightw;
        long plem_apos_rightpos = ((plem << 10) | apos) << 10 | rightpos;
        features[index++] = plem_apos_rightpos;
        long plem_apos_leftsiblingw = ((plem << 20) | leftsiblingw) << 10 | apos;
        features[index++] = plem_apos_leftsiblingw;
        long plem_apos_leftsiblingpos = ((plem << 10) | apos) << 10 | leftsiblingpos;
        features[index++] = plem_apos_leftsiblingpos;
        long plem_apos_rightsiblingw = ((plem << 20) | rightsiblingw) << 10 | apos;
        features[index++] = plem_apos_rightsiblingw;
        long plem_apos_rightsiblingpos = ((plem << 10) | apos) << 10 | rightsiblingpos;
        features[index++] = plem_apos_rightsiblingpos;
        String plem_adeprel_deprelpath = plem + " " + adeprel + " " + deprelpath;
        features[index++] = plem_adeprel_deprelpath;
        String plem_adeprel_pospath = plem + " " + adeprel + " " + pospath;
        features[index++] = plem_adeprel_pospath;
        long plem_adeprel_position = ((plem << 10) | adeprel) << 2 | position;
        features[index++] = plem_adeprel_position;
        long plem_adeprel_leftw = ((plem << 20) | leftw) << 10 | adeprel;
        features[index++] = plem_adeprel_leftw;
        long plem_adeprel_leftpos = ((plem << 10) | adeprel) << 10 | leftpos;
        features[index++] = plem_adeprel_leftpos;
        long plem_adeprel_rightw = ((plem << 20) | rightw) << 10 | adeprel;
        features[index++] = plem_adeprel_rightw;
        long plem_adeprel_rightpos = ((plem << 10) | adeprel) << 10 | rightpos;
        features[index++] = plem_adeprel_rightpos;
        long plem_adeprel_leftsiblingw = ((plem << 20) | leftsiblingw) << 10 | adeprel;
        features[index++] = plem_adeprel_leftsiblingw;
        long plem_adeprel_leftsiblingpos = ((plem << 10) | adeprel) << 10 | leftsiblingpos;
        features[index++] = plem_adeprel_leftsiblingpos;
        long plem_adeprel_rightsiblingw = ((plem << 20) | rightsiblingw) << 10 | adeprel;
        features[index++] = plem_adeprel_rightsiblingw;
        long plem_adeprel_rightsiblingpos = ((plem << 10) | adeprel) << 10 | rightsiblingpos;
        features[index++] = plem_adeprel_rightsiblingpos;
        String plem_deprelpath_pospath = plem + " " + deprelpath + " " + pospath;
        features[index++] = plem_deprelpath_pospath;
        String plem_deprelpath_position = plem + " " + deprelpath + " " + position;
        features[index++] = plem_deprelpath_position;
        String plem_deprelpath_leftw = plem + " " + leftw + " " + deprelpath;
        features[index++] = plem_deprelpath_leftw;
        String plem_deprelpath_leftpos = plem + " " + deprelpath + " " + leftpos;
        features[index++] = plem_deprelpath_leftpos;
        String plem_deprelpath_rightw = plem + " " + rightw + " " + deprelpath;
        features[index++] = plem_deprelpath_rightw;
        String plem_deprelpath_rightpos = plem + " " + deprelpath + " " + rightpos;
        features[index++] = plem_deprelpath_rightpos;
        String plem_deprelpath_leftsiblingw = plem + " " + leftsiblingw + " " + deprelpath;
        features[index++] = plem_deprelpath_leftsiblingw;
        String plem_deprelpath_leftsiblingpos = plem + " " + deprelpath + " " + leftsiblingpos;
        features[index++] = plem_deprelpath_leftsiblingpos;
        String plem_deprelpath_rightsiblingw = plem + " " + rightsiblingw + " " + deprelpath;
        features[index++] = plem_deprelpath_rightsiblingw;
        String plem_deprelpath_rightsiblingpos = plem + " " + deprelpath + " " + rightsiblingpos;
        features[index++] = plem_deprelpath_rightsiblingpos;
        String plem_pospath_position = plem + " " + pospath + " " + position;
        features[index++] = plem_pospath_position;
        String plem_pospath_leftw = plem + " " + leftw + " " + pospath;
        features[index++] = plem_pospath_leftw;
        String plem_pospath_leftpos = plem + " " + pospath + " " + leftpos;
        features[index++] = plem_pospath_leftpos;
        String plem_pospath_rightw = plem + " " + rightw + " " + pospath;
        features[index++] = plem_pospath_rightw;
        String plem_pospath_rightpos = plem + " " + pospath + " " + rightpos;
        features[index++] = plem_pospath_rightpos;
        String plem_pospath_leftsiblingw = plem + " " + leftsiblingw + " " + pospath;
        features[index++] = plem_pospath_leftsiblingw;
        String plem_pospath_leftsiblingpos = plem + " " + pospath + " " + leftsiblingpos;
        features[index++] = plem_pospath_leftsiblingpos;
        String plem_pospath_rightsiblingw = plem + " " + rightsiblingw + " " + pospath;
        features[index++] = plem_pospath_rightsiblingw;
        String plem_pospath_rightsiblingpos = plem + " " + pospath + " " + rightsiblingpos;
        features[index++] = plem_pospath_rightsiblingpos;
        long plem_position_leftw = ((plem << 20) | leftw) << 2 | position;
        features[index++] = plem_position_leftw;
        long plem_position_leftpos = ((plem << 10) | leftpos) << 2 | position;
        features[index++] = plem_position_leftpos;
        long plem_position_rightw = ((plem << 20) | rightw) << 2 | position;
        features[index++] = plem_position_rightw;
        long plem_position_rightpos = ((plem << 10) | rightpos) << 2 | position;
        features[index++] = plem_position_rightpos;
        long plem_position_leftsiblingw = ((plem << 20) | leftsiblingw) << 2 | position;
        features[index++] = plem_position_leftsiblingw;
        long plem_position_leftsiblingpos = ((plem << 10) | leftsiblingpos) << 2 | position;
        features[index++] = plem_position_leftsiblingpos;
        long plem_position_rightsiblingw = ((plem << 20) | rightsiblingw) << 2 | position;
        features[index++] = plem_position_rightsiblingw;
        long plem_position_rightsiblingpos = ((plem << 10) | rightsiblingpos) << 2 | position;
        features[index++] = plem_position_rightsiblingpos;
        long plem_leftw_leftpos = ((plem << 20) | leftw) << 10 | leftpos;
        features[index++] = plem_leftw_leftpos;
        long plem_leftw_rightw = ((plem << 20) | leftw) << 20 | rightw;
        features[index++] = plem_leftw_rightw;
        long plem_leftw_rightpos = ((plem << 20) | leftw) << 10 | rightpos;
        features[index++] = plem_leftw_rightpos;
        long plem_leftw_leftsiblingw = ((plem << 20) | leftw) << 20 | leftsiblingw;
        features[index++] = plem_leftw_leftsiblingw;
        long plem_leftw_leftsiblingpos = ((plem << 20) | leftw) << 10 | leftsiblingpos;
        features[index++] = plem_leftw_leftsiblingpos;
        long plem_leftw_rightsiblingw = ((plem << 20) | leftw) << 20 | rightsiblingw;
        features[index++] = plem_leftw_rightsiblingw;
        long plem_leftw_rightsiblingpos = ((plem << 20) | leftw) << 10 | rightsiblingpos;
        features[index++] = plem_leftw_rightsiblingpos;
        long plem_leftpos_rightw = ((plem << 20) | rightw) << 10 | leftpos;
        features[index++] = plem_leftpos_rightw;
        long plem_leftpos_rightpos = ((plem << 10) | leftpos) << 10 | rightpos;
        features[index++] = plem_leftpos_rightpos;
        long plem_leftpos_leftsiblingw = ((plem << 20) | leftsiblingw) << 10 | leftpos;
        features[index++] = plem_leftpos_leftsiblingw;
        long plem_leftpos_leftsiblingpos = ((plem << 10) | leftpos) << 10 | leftsiblingpos;
        features[index++] = plem_leftpos_leftsiblingpos;
        long plem_leftpos_rightsiblingw = ((plem << 20) | rightsiblingw) << 10 | leftpos;
        features[index++] = plem_leftpos_rightsiblingw;
        long plem_leftpos_rightsiblingpos = ((plem << 10) | leftpos) << 10 | rightsiblingpos;
        features[index++] = plem_leftpos_rightsiblingpos;
        long plem_rightw_rightpos = ((plem << 20) | rightw) << 10 | rightpos;
        features[index++] = plem_rightw_rightpos;
        long plem_rightw_leftsiblingw = ((plem << 20) | rightw) << 20 | leftsiblingw;
        features[index++] = plem_rightw_leftsiblingw;
        long plem_rightw_leftsiblingpos = ((plem << 20) | rightw) << 10 | leftsiblingpos;
        features[index++] = plem_rightw_leftsiblingpos;
        long plem_rightw_rightsiblingw = ((plem << 20) | rightw) << 20 | rightsiblingw;
        features[index++] = plem_rightw_rightsiblingw;
        long plem_rightw_rightsiblingpos = ((plem << 20) | rightw) << 10 | rightsiblingpos;
        features[index++] = plem_rightw_rightsiblingpos;
        long plem_rightpos_leftsiblingw = ((plem << 20) | leftsiblingw) << 10 | rightpos;
        features[index++] = plem_rightpos_leftsiblingw;
        long plem_rightpos_leftsiblingpos = ((plem << 10) | rightpos) << 10 | leftsiblingpos;
        features[index++] = plem_rightpos_leftsiblingpos;
        long plem_rightpos_rightsiblingw = ((plem << 20) | rightsiblingw) << 10 | rightpos;
        features[index++] = plem_rightpos_rightsiblingw;
        long plem_rightpos_rightsiblingpos = ((plem << 10) | rightpos) << 10 | rightsiblingpos;
        features[index++] = plem_rightpos_rightsiblingpos;
        long plem_leftsiblingw_leftsiblingpos = ((plem << 20) | leftsiblingw) << 10 | leftsiblingpos;
        features[index++] = plem_leftsiblingw_leftsiblingpos;
        long plem_leftsiblingw_rightsiblingw = ((plem << 20) | leftsiblingw) << 20 | rightsiblingw;
        features[index++] = plem_leftsiblingw_rightsiblingw;
        long plem_leftsiblingw_rightsiblingpos = ((plem << 20) | leftsiblingw) << 10 | rightsiblingpos;
        features[index++] = plem_leftsiblingw_rightsiblingpos;
        long plem_leftsiblingpos_rightsiblingw = ((plem << 20) | rightsiblingw) << 10 | leftsiblingpos;
        features[index++] = plem_leftsiblingpos_rightsiblingw;
        long plem_leftsiblingpos_rightsiblingpos = ((plem << 10) | rightsiblingpos) << 10 | leftsiblingpos;
        features[index++] = plem_leftsiblingpos_rightsiblingpos;
        long plem_rightSiblingw_rightSiblingpos = ((plem << 20) | rightsiblingw) << 10 | rightsiblingpos;
        features[index++] = plem_rightSiblingw_rightSiblingpos;
        //some miscellaneous tri-gram features
        int ppos_apos_adeprel = (((ppos << 10) | apos) << 10) | adeprel;
        features[index++] = ppos_apos_adeprel;
        int pdeprel_apos_adeprel = (((pdeprel << 10) | apos) << 10) | adeprel;
        features[index++] = pdeprel_apos_adeprel;
        String pchilddepset_apos_adeprel = pchilddepset + " " + apos + " " + adeprel;
        features[index++] = pchilddepset_apos_adeprel;
        String pchildposset_apos_adeprel = pchildposset + " " + apos + " " + adeprel;
        features[index++] = pchildposset_apos_adeprel;
        String pchildwset_apos_adeprel = pchildwset + " " + apos + " " + adeprel;
        features[index++] = pchildwset_apos_adeprel;
        String pchildwset_aw_adeprel = pchildwset + " " + aw + " " + adeprel;
        features[index++] = pchildwset_aw_adeprel;
        return new Object[]{features, index};
    }


    private static class BaseFeatureFields {
        private int pIdx;
        private int aIdx;
        private Sentence sentence;
        private IndexMap indexMap;
        private int pw;
        private int ppos;
        private int plem;
        private String pSense;
        private int pdeprel;
        private int pprw;
        private int pprpos;
        private String pdepsubcat;
        private String pchilddepset;
        private String pchildposset;
        private String pchildwset;
        private int aw;
        private int apos;
        private int adeprel;
        private String deprelpath;
        private String pospath;
        private int position;
        private int leftw;
        private int leftpos;
        private int rightw;
        private int rightpos;
        private int rightsiblingw;
        private int rightsiblingpos;
        private int leftsiblingw;
        private int leftsiblingpos;

        //word cluster features
        private int pwFullCluster;
        private int pw4cluster;
        private int plem_cluster;
        private int pprw_fullCluster;
        private int pprw_4cluster;
        private int aw_fullCluster;
        private int aw_4cluster;
        private int leftw_cluster;
        private int rightw_cluster;
        private int rightsiblingw_cluster;
        private int leftsiblingw_cluster;

        public BaseFeatureFields(int pIdx, int aIdx, Sentence sentence, IndexMap indexMap) {
            this.pIdx = pIdx;
            this.aIdx = aIdx;
            this.sentence = sentence;
            this.indexMap = indexMap;
        }

        public int getPw() {
            return pw;
        }

        public int getPpos() {
            return ppos;
        }

        public int getPlem() {
            return plem;
        }

        public String getpSense() {
            return pSense;
        }

        public int getPdeprel() {
            return pdeprel;
        }

        public int getPprw() {
            return pprw;
        }

        public int getPprpos() {
            return pprpos;
        }

        public String getPdepsubcat() {
            return pdepsubcat;
        }

        public String getPchilddepset() {
            return pchilddepset;
        }

        public String getPchildposset() {
            return pchildposset;
        }

        public String getPchildwset() {
            return pchildwset;
        }

        public int getAw() {
            return aw;
        }

        public int getApos() {
            return apos;
        }

        public int getAdeprel() {
            return adeprel;
        }

        public String getDeprelpath() {
            return deprelpath;
        }

        public String getPospath() {
            return pospath;
        }

        public int getPosition() {
            return position;
        }

        public int getLeftw() {
            return leftw;
        }

        public int getLeftpos() {
            return leftpos;
        }

        public int getRightw() {
            return rightw;
        }

        public int getRightpos() {
            return rightpos;
        }

        public int getRightsiblingw() {
            return rightsiblingw;
        }

        public int getRightsiblingpos() {
            return rightsiblingpos;
        }

        public int getLeftsiblingw() {
            return leftsiblingw;
        }

        public int getLeftsiblingpos() {
            return leftsiblingpos;
        }

        public int getPwFullCluster() {
            return pwFullCluster;
        }

        public int getPlem_cluster() {
            return plem_cluster;
        }

        public int getAw_fullCluster() {
            return aw_fullCluster;
        }

        public int getPprw_fullCluster() {
            return pprw_fullCluster;
        }

        public int getLeftw_cluster() {
            return leftw_cluster;
        }

        public int getRightw_cluster() {
            return rightw_cluster;
        }

        public int getRightsiblingw_cluster() {
            return rightsiblingw_cluster;
        }

        public int getLeftsiblingw_cluster() {
            return leftsiblingw_cluster;
        }

        public int getPw4cluster() {
            return pw4cluster;
        }

        public int getPprw_4cluster() {
            return pprw_4cluster;
        }

        public int getAw_4cluster() {
            return aw_4cluster;
        }

        public BaseFeatureFields invoke() throws Exception {
            int[] sentenceDepLabels = sentence.getDepLabels();
            int[] sentenceDepHeads = sentence.getDepHeads();
            int[] sentenceWords = sentence.getWords();
            int[] sentencePOSTags = sentence.getPosTags();
            int[] sentenceLemmas = sentence.getLemmas();
            int[] sentenceWordsFullClusterIds = sentence.getWordFullClusterIds();
            int[] sentenceWords4ClusterIds = sentence.getWord4ClusterIds();
            int[] sentenceLemmaClusterIds = sentence.getLemmaClusterIds();
            TreeSet<Integer>[] sentenceReverseDepHeads = sentence.getReverseDepHeads();
            HashMap<Integer, String> sentencePredicatesInfo = sentence.getPredicatesAutoLabelMap();

            //predicate features
            pw = sentenceWords[pIdx];
            pwFullCluster = sentenceWordsFullClusterIds[pIdx];
            pw4cluster = sentenceWords4ClusterIds[pIdx];
            ppos = sentencePOSTags[pIdx];
            plem = sentenceLemmas[pIdx];
            plem_cluster = sentenceLemmaClusterIds[pIdx];
            pSense = sentencePredicatesInfo.get(pIdx);
            pdeprel = sentenceDepLabels[pIdx];
            pprw = sentenceWords[sentenceDepHeads[pIdx]];
            pprw_fullCluster = sentenceWordsFullClusterIds[sentenceDepHeads[pIdx]];
            pprw_4cluster = sentenceWords4ClusterIds[sentenceDepHeads[pIdx]];
            pprpos = sentencePOSTags[sentenceDepHeads[pIdx]];
            pdepsubcat = getDepSubCat(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
            pchilddepset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceDepLabels, sentencePOSTags, indexMap);
            pchildposset = getChildSet(pIdx, sentenceReverseDepHeads, sentencePOSTags, sentencePOSTags, indexMap);
            pchildwset = getChildSet(pIdx, sentenceReverseDepHeads, sentenceWords, sentencePOSTags, indexMap);

            int leftMostDependentIndex = getLeftMostDependentIndex(aIdx, sentenceReverseDepHeads);
            int rightMostDependentIndex = getRightMostDependentIndex(aIdx, sentenceReverseDepHeads);
            int parIndex = sentenceDepHeads[aIdx];
            int lefSiblingIndex = getLeftSiblingIndex(aIdx, parIndex, sentenceReverseDepHeads);
            int rightSiblingIndex = getRightSiblingIndex(aIdx, parIndex, sentenceReverseDepHeads);

            //argument features
            aw = sentenceWords[aIdx];
            aw_fullCluster = sentenceWordsFullClusterIds[aIdx];
            aw_4cluster = sentenceWords4ClusterIds[aIdx];
            apos = sentencePOSTags[aIdx];
            adeprel = sentenceDepLabels[aIdx];

            //predicate-argument features
            deprelpath = StringUtils.convertPathArrayIntoString(sentence.getDepPath(pIdx, aIdx));
            pospath = StringUtils.convertPathArrayIntoString(sentence.getPOSPath(pIdx, aIdx));

            position = 0;
            if (pIdx < aIdx)
                position = 2; //after
            else if (pIdx > aIdx)
                position = 1; //before

            leftw = leftMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[leftMostDependentIndex];
            leftw_cluster = leftMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullClusterIdx : sentenceWordsFullClusterIds[leftMostDependentIndex];
            leftpos = leftMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[leftMostDependentIndex];
            rightw = rightMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[rightMostDependentIndex];
            rightw_cluster = rightMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullClusterIdx : sentenceWordsFullClusterIds[rightMostDependentIndex];
            rightpos = rightMostDependentIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[rightMostDependentIndex];
            rightsiblingw = rightSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[rightSiblingIndex];
            rightsiblingw_cluster = rightSiblingIndex == IndexMap.nullIdx ? IndexMap.nullClusterIdx : sentenceWordsFullClusterIds[rightSiblingIndex];
            rightsiblingpos = rightSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[rightSiblingIndex];
            leftsiblingw = lefSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentenceWords[lefSiblingIndex];
            leftsiblingw_cluster = lefSiblingIndex == IndexMap.nullIdx ? IndexMap.nullClusterIdx : sentenceWordsFullClusterIds[lefSiblingIndex];
            leftsiblingpos = lefSiblingIndex == IndexMap.nullIdx ? IndexMap.nullIdx : sentencePOSTags[lefSiblingIndex];
            return this;
        }
    }
}
