package SentenceStruct;

import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.PD.PD;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;

/**
 * Created by Maryam Aminian on 12/9/15.
 */
public class Sentence {

    private int[] depHeads;
    private int[] depLabels;
    private int[] words;
    private int[] wordClusterIds;
    private int[] posTags;
    private int[] cPosTags;
    private int[] lemmas;
    private int[] lemmaClusterIds;
    private String[] lemmas_str;
    private TreeSet<Integer>[] reverseDepHeads;
    private PAs predicateArguments;


    public Sentence(String sentence, IndexMap indexMap) {
        String[] tokens = sentence.trim().split("\n");

        int numTokens = tokens.length + 1; //add one more token for ROOT
        int predicatesSeq = -1;

        depHeads = new int[numTokens];
        depHeads[0] = IndexMap.nullIdx;
        depLabels = new int[numTokens];
        depLabels[0] = IndexMap.nullIdx;
        words = new int[numTokens];
        words[0] = indexMap.str2int("ROOT");
        posTags = new int[numTokens];
        posTags[0] = words[0];
        cPosTags = new int[numTokens];
        cPosTags[0] = words[0];
        lemmas = new int[numTokens];
        lemmas[0] = words[0];
        lemmas_str = new String[numTokens];
        lemmas_str[0] = "ROOT";
        wordClusterIds = new int[numTokens];
        wordClusterIds[0] = IndexMap.ROOTClusterIdx;
        lemmaClusterIds = new int[numTokens];
        lemmaClusterIds[0] = IndexMap.ROOTClusterIdx;

        reverseDepHeads = new TreeSet[numTokens];
        predicateArguments = new PAs();

        for (int tokenIdx = 0; tokenIdx < tokens.length; tokenIdx++) {
            String token = tokens[tokenIdx];
            String[] fields = token.split("\t");

            int index = Integer.parseInt(fields[0]);
            int depHead = Integer.parseInt(fields[9]);
            depHeads[index] = depHead;

            words[index] = indexMap.str2int(fields[1]);
            wordClusterIds[index] = indexMap.getClusterId(fields[1]);
            depLabels[index] = indexMap.str2int(fields[11]);
            posTags[index] = indexMap.str2int(fields[5]);
            cPosTags[index] = indexMap.str2int(util.StringUtils.getCoarsePOS(fields[5]));
            lemmas[index] = indexMap.str2int(fields[3]);
            lemmaClusterIds[index] = indexMap.getClusterId(fields[3]);

            if (reverseDepHeads[depHead] == null) {
                TreeSet<Integer> children = new TreeSet<Integer>();
                children.add(index);
                reverseDepHeads[depHead] = children;
            } else
                reverseDepHeads[depHead].add(index);

            String predicateGoldLabel = null;
            if (!fields[13].equals("_")) {
                predicatesSeq++;
                predicateGoldLabel = fields[13];
                predicateArguments.setPredicate(predicatesSeq, index, predicateGoldLabel);
            }

            if (fields.length > 14) //we have at least one argument
            {
                for (int i = 14; i < fields.length; i++) {
                    if (!fields[i].equals("_")) //found an argument
                    {
                        String argumentType = fields[i];
                        int associatedPredicateSeq = i - 14;
                        predicateArguments.setArgument(associatedPredicateSeq, index, argumentType);
                    }
                }
            }
        }
    }

    public ArrayList<Integer> getDepPath(int source, int target) {
        int right = 0;
        int left = 1;
        ArrayList<Integer> visited = new ArrayList<Integer>();

        if (source != target) {
            if (reverseDepHeads[source] != null) {
                //source has some children
                for (int child : reverseDepHeads[source]) {
                    if (child == target) {
                        if (child > source) {
                            visited.add(depLabels[child] << 1 | right);
                        } else {
                            visited.add(depLabels[child] << 1 | left);
                        }
                        break;
                    } else {
                        if (child > source) {
                            visited.add(depLabels[child] << 1 | right);
                        } else {
                            visited.add(depLabels[child] << 1 | left);
                        }
                        ArrayList<Integer> visitedFromThisChild = getDepPath(child, target);
                        if (visitedFromThisChild.size() != 0) {
                            visited.addAll(visitedFromThisChild);
                            break;
                        } else
                            visited.clear();
                    }
                }
            } else {
                //source does not have any children + we have not still met the target --> there is no path between source and target
                visited.clear();
            }
        }
        return visited;
    }

    public ArrayList<Integer> getPOSPath(int source, int target) {
        int right = 0;
        int left = 1;
        ArrayList<Integer> visited = new ArrayList<Integer>();

        if (source != target) {
            if (reverseDepHeads[source] != null) {
                //source has some children
                for (int child : reverseDepHeads[source]) {
                    if (child == target) {
                        if (child > source) {
                            visited.add(right);
                        } else {
                            visited.add(left);
                        }
                        break;
                    } else {
                        if (child > source) {
                            visited.add(posTags[child] << 1 | right);
                        } else {
                            visited.add(posTags[child] << 1 | left);
                        }
                        ArrayList<Integer> visitedFromThisChild = getPOSPath(child, target);
                        if (visitedFromThisChild.size() != 0) {
                            visited.addAll(visitedFromThisChild);
                            break;
                        } else
                            visited.clear();
                    }
                }
            } else {
                //source does not have any children + we have not still met the target --> there is no path between source and target
                visited.clear();
            }
        }
        return visited;
    }

    public PAs getPredicateArguments() {
        return predicateArguments;
    }

    public int[] getPosTags() {
        return posTags;
    }

    public int[] getcPosTags() {
        return cPosTags;
    }

    public int[] getDepHeads() {
        return depHeads;
    }

    public String[] getDepHeads_as_str() {
        String[] depHeads_str = new String[depHeads.length];
        for (int i = 0; i < depHeads.length; i++)
            depHeads_str[i] = Integer.toString(depHeads[i]);
        return depHeads_str;
    }

    public int[] getWords() {
        return words;
    }

    public int[] getDepLabels() {
        return depLabels;
    }

    public int[] getLemmas() {
        return lemmas;
    }

    public int[] getWordClusterIds() {
        return wordClusterIds;
    }

    public int[] getLemmaClusterIds() {
        return lemmaClusterIds;
    }

    public String[] getLemmas_str() {
        return lemmas_str;
    }

    public TreeSet<Integer>[] getReverseDepHeads() {
        return reverseDepHeads;
    }

    public HashMap<Integer, String> getPredicatesAutoLabelMap() {
        HashMap<Integer, String> predicatesInfo = new HashMap<Integer, String>();
        for (PA pa : predicateArguments.getPredicateArgumentsAsArray())
            predicatesInfo.put(pa.getPredicate().getIndex(), pa.getPredicate().getPredicateAutoLabel());

        return predicatesInfo;
    }

    public HashMap<Integer, String> getPredicatesGoldLabelMap() {
        HashMap<Integer, String> predicatesInfo = new HashMap<Integer, String>();
        for (PA pa : predicateArguments.getPredicateArgumentsAsArray())
            predicatesInfo.put(pa.getPredicate().getIndex(), pa.getPredicate().getPredicateGoldLabel());

        return predicatesInfo;
    }

    public ArrayList<Predicate> getPredicates () {
        ArrayList<Predicate> predicates = new ArrayList<>();
        for (PA pa: predicateArguments.getPredicateArgumentsAsArray())
            predicates.add(pa.getPredicate());
        return predicates;
    }

    public ArrayList<Integer> getPredicatesIndices() {
        ArrayList<Integer> predicateIndices = new ArrayList<>();
        for (PA pa: predicateArguments.getPredicateArgumentsAsArray())
            predicateIndices.add(pa.getPredicate().getIndex());
        return predicateIndices;
    }

    public void setPDAutoLabels (HashMap<Integer, String> pdAutoLabels){
        assert pdAutoLabels.size() == predicateArguments.getPredicateArgumentsAsArray().size();
        for (PA pa: predicateArguments.getPredicateArgumentsAsArray()){
            int pIdx = pa.getPredicate().getIndex();
            assert pdAutoLabels.containsKey(pIdx);
            pa.getPredicate().setPredicateAutoLabel(pdAutoLabels.get(pIdx));
        }
    }
}
