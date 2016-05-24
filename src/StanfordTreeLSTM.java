import Sentence.Argument;
import Sentence.PA;
import Sentence.Sentence;

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

    static HashSet<String> vocab= new HashSet<String>();

    public static ArrayList<String> generateData4StanfordTreeLSTM (Sentence sen, boolean justCoreRoles) {

        ArrayList<PA> pas_list = sen.getPredicateArguments().getPredicateArgumentsAsArray();
        String[] posTags = sen.getPosTags();
        String[] words= sen.getWords();
        String[] depRels = sen.getDepLabels();
        String[] depParents= sen.getDepHeads_as_str();

        String sentences2write="";
        String depRels2write="";
        String depParents2write="";
        String depLabels2write="";

        if (pas_list.size()>0)
        {
            //sentence has at least one predicate-argument
            for (PA pa : pas_list) {

                int predicateIndex = pa.getPredicateIndex();

                sentences2write+= String.join(" ", Arrays.copyOfRange(words, 1, words.length)).trim()+"\n";
                depRels2write+= String.join(" ",Arrays.copyOfRange(depRels, 1, depRels.length)).trim()+"\n";
                depParents2write+= String.join(" ", Arrays.copyOfRange(depParents, 1, depParents.length)).trim()+"\n";

                if (posTags[predicateIndex].startsWith("VB")) {

                    String srl_label="";
                    String test_label="";
                    ArrayList<Argument> arguments = pa.getArguments();
                    HashMap<Integer,String> argumentsInfo= extractArgumentInfo(arguments, justCoreRoles);

                    for (int wordIndex=0; wordIndex< words.length; wordIndex++) {
                        if (wordIndex== predicateIndex)
                        {
                            //seen the predicate
                            //srl_label+= "p ";
                            srl_label+= "-1 ";//predicate will be shawn with "-1"
                            //srl_label+= "-2 ";
                        }
                        else
                        {
                            //seen the argument
                            if (argumentsInfo.containsKey(wordIndex)) {
                                //this word has the desired argument type
                                //srl_label+= argumentsInfo.get(wordIndex)+" ";
                                //test_label = test_5class_labels_to_run_treeLSTM(argumentsInfo.get(wordIndex));
                                //srl_label+= test_label+" ";
                                srl_label+= argumentsInfo.get(wordIndex).substring(1,argumentsInfo.get(wordIndex).length())+" "; //add argument index as the label

                            }
                            else
                                //srl_label+= "0 ";
                                srl_label+= "0 "; //neutral
                        }
                    }
                    depLabels2write+= String.join(" ",Arrays.copyOfRange(srl_label.split(" "),1,words.length)).trim()+"\n";
                }else
                {
                    String srl_label="";
                    for (int wordIndex=0; wordIndex< words.length; wordIndex++)
                        srl_label+= "0 ";
                    depLabels2write+= String.join(" ",Arrays.copyOfRange(srl_label.split(" "),1,words.length)).trim()+"\n";
                }
            }
        }else
        {
            //sentence does not have any predicate
            String srl_label="";
            for (int wordIndex=0; wordIndex< words.length; wordIndex++)
                srl_label+= "0 ";

            depLabels2write+= String.join(" ",Arrays.copyOfRange(srl_label.split(" "),1,words.length)).trim()+"\n";
            sentences2write+= String.join(" ", Arrays.copyOfRange(words, 1, words.length)).trim()+"\n";
            depRels2write+= String.join(" ",Arrays.copyOfRange(depRels, 1, depRels.length)).trim()+"\n";
            depParents2write+= String.join(" ", Arrays.copyOfRange(depParents, 1, depParents.length)).trim()+"\n";
        }


        ArrayList<String> writables= new ArrayList<String>();
        writables.add(sentences2write.trim());
        writables.add(depParents2write.trim());
        writables.add(depRels2write.trim());
        writables.add(depLabels2write.trim());

        return writables;
    }


    public static HashMap<Integer, String> extractArgumentInfo (ArrayList<Argument> args, boolean justCoreSemanticRoles){
        HashMap<Integer, String> argsInfo= new HashMap<Integer, String>();
        for (Argument arg: args)
        {
            int arg_index= arg.getIndex();
            final String arg_type= arg.getType();

            if (justCoreSemanticRoles && isACoreRole(arg_type)){
                if (!argsInfo.containsKey(arg_index))
                    argsInfo.put(arg_index, arg_type);
                else
                    System.out.print("Some thing's wrong! one word with two labels!");
            }
            else if(!justCoreSemanticRoles){
                if (!argsInfo.containsKey(arg_index))
                    argsInfo.put(arg_index, arg_type);
                else
                    System.out.print("Some thing's wrong! one word with two labels!");
            }
        }
        return argsInfo;
    }


    public static boolean isACoreRole(String role){
        if(role.startsWith("A0")|| role.startsWith("A1") || role.startsWith("A2")||
                role.startsWith("A3")|| role.startsWith("A4")|| role.startsWith("A5")||
                role.startsWith("A6")||role.startsWith("A7") || role.startsWith("A8") ||
                role.startsWith("A9"))
            return true;
        else
            return false;
    }

    public static void updateVocab (Sentence sen){
        String[] words= Arrays.copyOfRange(sen.getWords(), 1,sen.getWords().length);
        for (String word:words)
        {
            if (!vocab.contains(word))
                vocab.add(word);
        }
    }

    public static void writeVocab(String filePath, boolean cased) throws IOException{
        BufferedWriter treeLSTM_vocab_writer= new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filePath)));
        for (String word: vocab)
        {
            if (cased)
                treeLSTM_vocab_writer.write(word+"\n");
            else
            {
                String word_cased= word.toLowerCase();
                treeLSTM_vocab_writer.write(word_cased + "\n");
            }
        }

        treeLSTM_vocab_writer.flush();
        treeLSTM_vocab_writer.close();
    }

    public static String test_5class_labels_to_run_treeLSTM(String input)
    {
        String output="0";
        if (input.equals("A1"))
            output="-1";
        else if (input.equals("A2"))
            output="1";
        else if (input.equals("A3"))
            output="2";

        return output;

    }

}
