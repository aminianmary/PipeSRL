package Sentence;

import apple.laf.JRSUIUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeSet;

/**
 * Created by monadiab on 12/9/15.
 */
public class Sentence {

    int[] depHeads;
    String[] depLabels;
    String[] words;
    String[] posTags;
    String[] feats;
    String[] lemmas;
    HashMap<Integer, TreeSet<Integer>> reverseDepHeads;
    PAs predicateArguments;

    public Sentence(String sentence, String sentence_GD) {
        String[] tokens = sentence.trim().split("\n");
        String[] tokens_GD = sentence_GD.trim().split("\n");

        int numTokens = tokens.length + 1; //add one more token to account for ROOT
        int predicatesSeq = -1;

        depHeads = new int[numTokens];
        depLabels = new String[numTokens];
        words = new String[numTokens];
        posTags = new String[numTokens];
        feats = new String[numTokens];
        lemmas = new String[numTokens];
        reverseDepHeads = new HashMap<Integer, TreeSet<Integer>>();
        predicateArguments = new PAs();

        for (int tokenIdx = 0; tokenIdx < tokens.length; tokenIdx++) {
            String token = tokens[tokenIdx];
            String token_GD = tokens_GD[tokenIdx];

            String[] fields = token.split("\t");
            String[] fields_GD = token_GD.split("\t");

            final int index = Integer.parseInt(fields[0]);
            final int depHead = Integer.parseInt(fields_GD[6]);

            words[index] = fields[1]; //isSimilarToLabelSet(fields[1]) ? "NUM": fields[1];
            depHeads[index] = depHead;
            depLabels[index] = fields_GD[7];
            posTags[index] = fields_GD[3];
            feats[index] = fields[7];
            lemmas[index] = fields[3];

            if (!reverseDepHeads.containsKey(depHead))
                reverseDepHeads.put(depHead, new TreeSet<Integer>() {{
                    add(index);
                }});
            else
                reverseDepHeads.get(depHead).add(index);


            //setPredicate predicate information
            String predicate = "-";
            if (!fields[13].equals("_")) {
                predicatesSeq++;
                predicate = fields[13];
                predicateArguments.setPredicate(predicatesSeq, index, predicate);
            }
            //setPredicate argument information
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


    public Sentence(String sentence) {
        String[] tokens = sentence.trim().split("\n");

        int numTokens = tokens.length + 1; //add one more token to account for ROOT
        int predicatesSeq = -1;

        depHeads = new int[numTokens];
        depLabels = new String[numTokens];
        words = new String[numTokens];
        posTags = new String[numTokens];
        feats = new String[numTokens];
        lemmas = new String[numTokens];
        reverseDepHeads = new HashMap<Integer, TreeSet<Integer>>();
        predicateArguments = new PAs();

        for (int tokenIdx = 0; tokenIdx < tokens.length; tokenIdx++) {
            String token = tokens[tokenIdx];
            String[] fields = token.split("\t");

            final int index = Integer.parseInt(fields[0]);
            final int depHead = Integer.parseInt(fields[9]);

            words[index] = fields[1]; //isSimilarToLabelSet(fields[1]) ? "NUM": fields[1];
            depHeads[index] = depHead;
            depLabels[index] = fields[11];
            posTags[index] = fields[5];
            feats[index] = fields[7];
            lemmas[index] = fields[3];

            if (!reverseDepHeads.containsKey(depHead))
                reverseDepHeads.put(depHead, new TreeSet<Integer>() {{
                    add(index);
                }});
            else
                reverseDepHeads.get(depHead).add(index);


            //setPredicate predicate information
            String predicate = "-";
            if (!fields[13].equals("_")) {
                predicatesSeq++;
                predicate = fields[13];
                predicateArguments.setPredicate(predicatesSeq, index, predicate);
            }
            //setPredicate argument information
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


    public TreeSet<String> getDepPath(int source, int target) {
        TreeSet<String> visited = new TreeSet<String>();

        if (reverseDepHeads.containsKey(source) && reverseDepHeads.get(source).size() > 0) {
            for (int child : reverseDepHeads.get(source)) {
                if (child == target) {
                    if (child > source)
                        visited.add(depLabels[child]+"_L");
                    else
                        visited.add(depLabels[child]+"_R");
                    break;
                }
                else {
                    if (child > source)
                        visited.add(depLabels[child] + "_L");
                    else
                        visited.add(depLabels[child] + "_R");

                    getDepPath(child, target);
                }
            }

        }
        return visited;
    }


    public TreeSet<String> getPOSPath(int source, int target) {
        TreeSet<String> visited = new TreeSet<String>();

        if (reverseDepHeads.containsKey(source) && reverseDepHeads.get(source).size() > 0) {
            for (int child : reverseDepHeads.get(source)) {
                if (child == target) {
                    if (child > source)
                        visited.add(posTags[child]+"_L");
                    else
                        visited.add(posTags[child]+"_R");
                    break;
                }
                else {
                    if (child > source)
                        visited.add(posTags[child]+"_L");
                    else
                        visited.add(posTags[child]+"_R");

                    getDepPath(child, target);
                }
            }

        }
        return visited;
    }



    public PAs getPredicateArguments() {
        return predicateArguments;
    }


    public String[] getPosTags() {
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

    public String[] getWords() {
        return words;
    }

    public String[] getDepLabels() {
        return depLabels;
    }

    public String[] getFeats() {
        return feats;
    }

    public String[] getLemmas() {
        return lemmas;
    }

    public HashMap<Integer, TreeSet<Integer>> getReverseDepHeads() {
        return reverseDepHeads;
    }

    public boolean isSimilarToLabelSet(String input) {
        boolean isSimilarToLabelSet = false;

        if (input.equals("1") || input.equals("2")
                || input.equals("3") || input.equals("4")
                || input.equals("5") || input.equals("6")
                || input.equals("7"))
            isSimilarToLabelSet = true;

        return isSimilarToLabelSet;
    }

    public String getVoice(int pIndex) {
        String voice = "a";
        if (reverseDepHeads.containsKey(pIndex) && reverseDepHeads.get(pIndex).size() > 0) {
            for (int child : reverseDepHeads.get(pIndex)) {
                if (depLabels[child].endsWith("pass"))
                    voice = "p";
            }
        }

        return voice;
    }

    public ArrayList<Predicate> getListOfPredicates()
    {
        ArrayList<Predicate> predicates= new ArrayList<Predicate>();
        for (PA pa: predicateArguments.getPredicateArgumentsAsArray())
            predicates.add(pa.getPredicate());

        return predicates;
    }
}
