package Sentence;

import SupervisedSRL.Strcutures.IndexMap;
import apple.laf.JRSUIUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;

/**
 * Created by Maryam Aminian on 12/9/15.
 */
public class Sentence {

    int[] depHeads;
    int[] depLabels;
    int[] words;
    int[] posTags;
    int[] cPosTags;
    int[] lemmas;
    String[] lemmas_str;
    TreeSet<Integer>[] reverseDepHeads;
    PAs predicateArguments;


    public Sentence(String sentence, IndexMap indexMap, boolean decode) {
        HashMap<String, Integer> wordMap = indexMap.getString2intMap();
        String[] tokens = sentence.trim().split("\n");

        int numTokens = tokens.length + 1; //add one more token for ROOT
        int predicatesSeq = -1;

        depHeads = new int[numTokens];
        depHeads[0] = indexMap.getNullIdx();
        depLabels = new int[numTokens];
        depLabels[0] = indexMap.getNullIdx();
        words = new int[numTokens];
        words[0] = indexMap.str2int("ROOT");
        posTags = new int[numTokens];
        posTags[0] = words[0];
        cPosTags = new int[numTokens];
        cPosTags[0] = words[0];
        lemmas = new int[numTokens];
        lemmas[0] = words[0];
        lemmas_str= new String[numTokens];
        lemmas_str[0] = "ROOT";

        reverseDepHeads = new TreeSet [numTokens];
        predicateArguments = new PAs();

        for (int tokenIdx = 0; tokenIdx < tokens.length; tokenIdx++) {
            String token = tokens[tokenIdx];
            String[] fields = token.split("\t");

            int index = Integer.parseInt(fields[0]);
            int depHead = Integer.parseInt(fields[9]);
            depHeads[index] = depHead;

            if (decode==false) {
                words[index] = wordMap.get(fields[1]);
                depLabels[index] = wordMap.get(fields[11]);
                posTags[index] = wordMap.get(fields[5]);
                cPosTags[index] = wordMap.get(util.StringUtils.getCoarsePOS(fields[5]));
                lemmas[index] = wordMap.get(fields[3]);
            }
            else
            {
                lemmas_str[index] = fields[3];

                if (wordMap.containsKey(fields[1]))
                    words[index] = wordMap.get(fields[1]);
                else
                    words[index] = indexMap.getUnknownIdx();

                if (wordMap.containsKey(fields[11]))
                    depLabels[index] = wordMap.get(fields[11]);
                else
                    depLabels[index] = indexMap.getUnknownIdx();

                if (wordMap.containsKey(fields[5]))
                    posTags[index] = wordMap.get(fields[5]);
                else
                    posTags[index] = indexMap.getUnknownIdx();

                if (wordMap.containsKey(util.StringUtils.getCoarsePOS(fields[5])))
                    cPosTags[index] = wordMap.get(util.StringUtils.getCoarsePOS(fields[5]));
                else
                    cPosTags[index] = indexMap.getUnknownIdx();

                if (wordMap.containsKey(fields[3]))
                    lemmas[index] = wordMap.get(fields[3]);
                else
                    lemmas[index] = indexMap.getUnknownIdx();
            }

            if (reverseDepHeads[depHead] == null){
                TreeSet<Integer> children= new TreeSet<Integer>();
                children.add(index);
                reverseDepHeads[depHead] = children;

            } else
                reverseDepHeads[depHead].add(index);

            //setPredicate predicate information
            String predicate = "_";
            if (!fields[13].equals("_")) {
                predicatesSeq++;
                predicate = fields[13];
                predicateArguments.setPredicate(predicatesSeq, index, predicate);
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
                        }
                        else
                            visited.clear();
                    }
                }
            }else
            {
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
                        }
                        else
                            visited.clear();
                    }
                }
            }else
            {
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


    public int[] getCPosTags() {
        return cPosTags;
    }

    public int[] getDepHeads() {
        return depHeads;
    }

    public String[] getLemmas_str() {return lemmas_str;}

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

    public TreeSet<Integer>[] getReverseDepHeads() {
        return reverseDepHeads;
    }

    public ArrayList<Predicate> getListOfPredicates()
    {
        ArrayList<Predicate> predicates= new ArrayList<Predicate>();
        for (PA pa: predicateArguments.getPredicateArgumentsAsArray())
            predicates.add(pa.getPredicate());

        return predicates;
    }
}
