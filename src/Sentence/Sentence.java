package Sentence;

import SupervisedSRL.Strcutures.IndexMap;

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
    int[] lemmas;
    TreeSet<Integer>[] reverseDepHeads;
    PAs predicateArguments;


    public Sentence(String sentence, IndexMap indexMap, boolean decode) {
        HashMap<String, Integer> wordMap = indexMap.getString2intMap();
        String[] tokens = sentence.trim().split("\n");

        int numTokens = tokens.length + 1; //add one more token for ROOT
        int predicatesSeq = -1;

        depHeads = new int[numTokens];
        depLabels = new int[numTokens];
        words = new int[numTokens];
        posTags = new int[numTokens];
        lemmas = new int[numTokens];

        reverseDepHeads = new TreeSet[numTokens];
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
                lemmas[index] = wordMap.get(fields[3]);
            }else
            {
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


    // todo be a member for Predicate (L/R show as bit, POS as int; finally with a StringBuilder concat them with space)
    public ArrayList<Integer> getDepPath(int source, int target) {
        int right = 0;
        int left = 1;
        ArrayList<Integer> visited = new ArrayList<Integer>();

        if (reverseDepHeads[source] != null) {
            for (int child : reverseDepHeads[source]) {
                if (child == target) {
                    if (child > source)
                        visited.add(depLabels[child]<<1 | left);
                    else
                        visited.add(depLabels[child]<<1 | right);
                    break;
                }
                else {
                    if (child > source)
                        visited.add(depLabels[child]<<1 | left);
                    else
                        visited.add(depLabels[child]<<1 | right);

                    getDepPath(child, target);
                }
            }

        }
        return visited;
    }


    // todo be a member for Predicate (L/R show as bit, POS as int; finally with a StringBuilder concat them with space)
    public ArrayList<Integer> getPOSPath(int source, int target) {
        int right = 0;
        int left = 1;
        ArrayList<Integer> visited = new ArrayList<Integer>();

        if (reverseDepHeads[source] != null) {
            for (int child : reverseDepHeads[source]) {
                if (child == target) {
                    if (child > source)
                        visited.add(posTags[child]<<1 | left);
                    else
                        visited.add(posTags[child]<<1 | right);
                    break;
                }
                else {
                    if (child > source)
                        visited.add(posTags[child]<<1 | left);
                    else
                        visited.add(posTags[child]<<1 | right);

                    getPOSPath(child, target);
                }
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
