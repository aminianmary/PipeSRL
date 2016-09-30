import SentenceStruct.Argument;
import SentenceStruct.PA;
import SentenceStruct.Sentence;
import SupervisedSRL.Strcutures.IndexMap;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;


/**
 * Created by monadiab on 12/21/15.
 */
public class StanfordTreeLSTM {

    static HashSet<Integer> vocab = new HashSet<Integer>();

    public static ArrayList<String> generateData4StanfordTreeLSTM(Sentence sen, IndexMap indexMap, boolean justCoreRoles) throws Exception {

        ArrayList<PA> pas_list = sen.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] posTags = sen.getPosTags();
        int[] words = sen.getWords();
        int[] depRels = sen.getDepLabels();
        String[] depParents = sen.getDepHeads_as_str();

        StringBuilder sentences2write = new StringBuilder();
        StringBuilder depRels2write = new StringBuilder();
        StringBuilder depParents2write = new StringBuilder();
        StringBuilder depLabels2write = new StringBuilder();

        if (pas_list.size() > 0) {
            //sentence has at least one predicate-argument
            for (PA pa : pas_list) {

                int predicateIndex = pa.getPredicate().getIndex();
                for (int k = 1; k < words.length; k++)
                    sentences2write.append(words[k] + " ");
                for (int k = 1; k < depRels.length; k++)
                    depRels2write.append(depRels[k] + " ");
                for (int k = 1; k < depParents.length; k++)
                    depParents2write.append(depParents[k] + " ");

                if (indexMap.int2str(posTags[predicateIndex]).startsWith("VB")) {

                    String srl_label = "";
                    String test_label = "";
                    ArrayList<Argument> arguments = pa.getArguments();
                    HashMap<Integer, String> argumentsInfo = extractArgumentInfo(arguments, justCoreRoles);

                    for (int wordIndex = 0; wordIndex < words.length; wordIndex++) {
                        if (wordIndex == predicateIndex) {
                            //seen the predicate
                            //srl_label+= "p ";
                            srl_label += "-1 ";//predicate will be shawn with "-1"
                            //srl_label+= "-2 ";
                        } else {
                            //seen the argument
                            if (argumentsInfo.containsKey(wordIndex)) {
                                //this word has the desired argument type
                                //srl_label+= argumentsInfo.get(wordIndex)+" ";
                                //test_label = test_5class_labels_to_run_treeLSTM(argumentsInfo.get(wordIndex));
                                //srl_label+= test_label+" ";
                                srl_label += argumentsInfo.get(wordIndex).substring(1, argumentsInfo.get(wordIndex).length()) + " "; //add argument index as the label

                            } else
                                //srl_label+= "0 ";
                                srl_label += "0 "; //neutral
                        }
                    }

                    for (int k = 1; k < words.length; k++)
                        depLabels2write.append(srl_label.split(" ")[k] + " ");

                } else {
                    String srl_label = "";
                    for (int wordIndex = 0; wordIndex < words.length; wordIndex++)
                        srl_label += "0 ";

                    for (int k = 1; k < words.length; k++)
                        depLabels2write.append(srl_label.split(" ")[k] + " ");

                }
            }
        } else {
            //sentence does not have any predicate
            String srl_label = "";
            for (int wordIndex = 0; wordIndex < words.length; wordIndex++)
                srl_label += "0 ";

            for (int k = 1; k < words.length; k++)
                sentences2write.append(words[k] + " ");
            for (int k = 1; k < depRels.length; k++)
                depRels2write.append(depRels[k] + " ");
            for (int k = 1; k < depParents.length; k++)
                depParents2write.append(depParents[k] + " ");
            for (int k = 1; k < words.length; k++)
                depLabels2write.append(srl_label.split(" ")[k] + " ");

        }


        ArrayList<String> writables = new ArrayList<String>();
        writables.add(sentences2write.toString().trim());
        writables.add(depParents2write.toString().trim());
        writables.add(depRels2write.toString().trim());
        writables.add(depLabels2write.toString().trim());

        return writables;
    }


    public static HashMap<Integer, String> extractArgumentInfo(ArrayList<Argument> args, boolean justCoreSemanticRoles) {
        HashMap<Integer, String> argsInfo = new HashMap<Integer, String>();
        for (Argument arg : args) {
            int arg_index = arg.getIndex();
            final String arg_type = arg.getType();

            if (justCoreSemanticRoles && isACoreRole(arg_type)) {
                if (!argsInfo.containsKey(arg_index))
                    argsInfo.put(arg_index, arg_type);
                else
                    System.out.print("Some thing's wrong! one word with two labels!");
            } else if (!justCoreSemanticRoles) {
                if (!argsInfo.containsKey(arg_index))
                    argsInfo.put(arg_index, arg_type);
                else
                    System.out.print("Some thing's wrong! one word with two labels!");
            }
        }
        return argsInfo;
    }


    public static boolean isACoreRole(String role) {
        if (role.startsWith("A0") || role.startsWith("A1") || role.startsWith("A2") ||
                role.startsWith("A3") || role.startsWith("A4") || role.startsWith("A5") ||
                role.startsWith("A6") || role.startsWith("A7") || role.startsWith("A8") ||
                role.startsWith("A9"))
            return true;
        else
            return false;
    }

    public static void updateVocab(Sentence sen) {
        int[] words = Arrays.copyOfRange(sen.getWords(), 1, sen.getWords().length);
        for (int word : words) {
            if (!vocab.contains(word))
                vocab.add(word);
        }
    }

    public static void writeVocab(String filePath /*, boolean cased*/) throws IOException {
        BufferedWriter treeLSTM_vocab_writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filePath)));
        for (int word : vocab) {
            /*if (cased)*/
            treeLSTM_vocab_writer.write(word + "\n");
            /*else
            {
                String word_cased= word.toLowerCase();
                treeLSTM_vocab_writer.write(word_cased + "\n");
            }*/
        }

        treeLSTM_vocab_writer.flush();
        treeLSTM_vocab_writer.close();
    }

    public static String test_5class_labels_to_run_treeLSTM(String input) {
        String output = "0";
        if (input.equals("A1"))
            output = "-1";
        else if (input.equals("A2"))
            output = "1";
        else if (input.equals("A3"))
            output = "2";

        return output;

    }

}
