import SentenceStruct.Argument;
import SentenceStruct.ArgumentPosition;
import SentenceStruct.PA;
import SentenceStruct.Sentence;
import SupervisedSRL.Strcutures.IndexMap;
import util.IO;

import java.io.*;
import java.util.*;

/**
 * Created by Maryam Aminian on 12/10/15.
 */
public class extract_argument_combination_classes {

    //static HashMap<String, ArrayList<Integer>> predArgLabelFreqDic= new HashMap<String, ArrayList<Integer>>();
    static HashMap<HashSet<String>, ArrayList<Integer>> predArgLabelFreqDic = new HashMap<HashSet<String>, ArrayList<Integer>>();
    static ArrayList<String> sentences = new ArrayList<String>();

    public static void main(String[] args) throws Exception {
        String propBankFile = args[0];
        String output_dir_path = args[1];
        boolean justCoreRoles = Boolean.parseBoolean(args[2]);
        String clusterFilePath = args[3];

        final IndexMap indexMap = new IndexMap(propBankFile, clusterFilePath);

        //output files
        String predArgLabelFile = output_dir_path + "predArgLabel.out";
        String treeLSTM_sentences = output_dir_path + "sents.txt";
        String treeLSTM_sentences_tok = output_dir_path + "sents.toks";
        String treeLSTM_dep_parents = output_dir_path + "dparents.txt";
        String treeLSTM_dep_rels = output_dir_path + "rels.txt";
        String treeLSTM_dep_labels = output_dir_path + "dlabels.txt";
        String treeLSTM_vocab = output_dir_path + "vocab.txt";
        String treeLSTM_vocab_cased = output_dir_path + "vocab-cased.txt";

        //output streams
        BufferedWriter treeLSTM_sentences_writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(treeLSTM_sentences)));
        BufferedWriter treeLSTM_sentences_tok_writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(treeLSTM_sentences_tok)));
        BufferedWriter treeLSTM_dep_parents_writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(treeLSTM_dep_parents)));
        BufferedWriter treeLSTM_dep_rels_writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(treeLSTM_dep_rels)));
        BufferedWriter treeLSTM_dep_labels_writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(treeLSTM_dep_labels)));


        BufferedReader propBankReader = new BufferedReader(new InputStreamReader(new FileInputStream(propBankFile)));

        String line2Read = "";
        String sentence = "";
        int sentenceCounter = -1;

        while ((line2Read = propBankReader.readLine()) != null) {
            line2Read = line2Read.trim();
            if (line2Read.equals("")) //sentence break
            {
                sentenceCounter++;
                if (sentenceCounter % 100 == 0)
                    System.out.println(sentenceCounter);

                boolean decode = false;
                Sentence sen = new Sentence(sentence, indexMap);

                //extracts data for treeLSTM
                ArrayList<String> treeLSTM_format_sentence = StanfordTreeLSTM.generateData4StanfordTreeLSTM(sen,
                        indexMap, justCoreRoles);
                StanfordTreeLSTM.updateVocab(sen);

                treeLSTM_sentences_writer.write(treeLSTM_format_sentence.get(0) + "\n");
                treeLSTM_sentences_tok_writer.write(treeLSTM_format_sentence.get(0) + "\n");
                treeLSTM_dep_parents_writer.write(treeLSTM_format_sentence.get(1) + "\n");
                treeLSTM_dep_rels_writer.write(treeLSTM_format_sentence.get(2) + "\n");
                treeLSTM_dep_labels_writer.write(treeLSTM_format_sentence.get(3) + "\n");
                StanfordTreeLSTM.writeVocab(treeLSTM_vocab /*, false*/);
                StanfordTreeLSTM.writeVocab(treeLSTM_vocab_cased /*, true*/);

                //keep the sentence for tracking later
                sentences.add(sentence);

                //ArrayList<String> labels =extractPredicateArgumentLabel(sen, true, true);
                ArrayList<HashSet<String>> labels = extractPredicateArgumentLabel_unorderedFormat(sen, indexMap, false, true);
                //updatePredArgLabelFreqDic(labels, sentenceCounter);
                updatePredArgLabelFreqDic_unorderedFormat(labels, sentenceCounter);

                sentence = "";
            } else {
                sentence += line2Read + "\n";
            }

        }

        //printLabelFreq(predArgLabelFile);
        printLabelFreq_unorderedFormat(predArgLabelFile);

        treeLSTM_sentences_writer.flush();
        treeLSTM_sentences_writer.close();

        treeLSTM_sentences_tok_writer.flush();
        treeLSTM_sentences_tok_writer.close();

        treeLSTM_dep_parents_writer.flush();
        treeLSTM_dep_parents_writer.close();

        treeLSTM_dep_rels_writer.flush();
        treeLSTM_dep_rels_writer.close();

        treeLSTM_dep_labels_writer.flush();
        treeLSTM_dep_labels_writer.close();

    }

    public static ArrayList<String> extractPredicateArgumentLabel(Sentence sen,
                                                                  IndexMap indexMap,
                                                                  boolean justMainPredicate,
                                                                  boolean justCoreRoles) throws Exception {
        ArrayList<String> argumentLabels = new ArrayList<String>();
        ArrayList<PA> pas = sen.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] posTags = sen.getPosTags();

        if (justMainPredicate == true) {
            boolean seenTheMainPredicate = false;
            for (PA pa : pas) {
                String label = "";
                boolean firstArgAfterPredicate = true;
                int predicateIndex = pa.getPredicateIndex();
                int predicateHeadIndex = sen.getDepHeads()[predicateIndex];

                if (predicateHeadIndex == 0) {
                    //make sure predicate is a verb (conll08 data contains nominal predicates from NomaBank too)

                    if (seenTheMainPredicate == true)
                        System.out.println("There are two predicates in this sentence with ROOT as their head!");

                    seenTheMainPredicate = true;

                    if (indexMap.int2str(posTags[predicateIndex]).startsWith("VB")) {

                        ArrayList<Argument> arguments = pa.getArguments();
                        boolean isAnyArgumentSeenAfterPredicate = isAnyArgumentSeenAfterPredicate(arguments);

                        if (isAnyArgumentSeenAfterPredicate == true) {
                            for (Argument ar : arguments) {

                                if (justCoreRoles == true) {
                                    if (isACoreRole(ar.getType()) == true) {
                                        if (ar.getArgPosition() == ArgumentPosition.AFTER && firstArgAfterPredicate == true) {
                                            label += "p|" + ar.getType() + "|";
                                            firstArgAfterPredicate = false;
                                        } else
                                            label += ar.getType() + "|";
                                    }

                                } else {
                                    if (ar.getArgPosition() == ArgumentPosition.AFTER && firstArgAfterPredicate == true) {
                                        label += "p|" + ar.getType() + "|";
                                        firstArgAfterPredicate = false;
                                    } else
                                        label += ar.getType() + "|";
                                }
                            }
                        } else {
                            for (Argument ar : arguments) {

                                if (justCoreRoles == true) {
                                    if (isACoreRole(ar.getType()) == true) {
                                        label += ar.getType() + "|";
                                    }

                                } else {
                                    label += ar.getType() + "|";
                                }
                            }
                        }


                        if (!label.equals("")) {
                            if (isAnyArgumentSeenAfterPredicate == true)
                                argumentLabels.add(label.substring(0, label.length() - 1));
                            else
                                argumentLabels.add(label + "p");
                        }
                    }
                }

            }

        } else {
            for (PA pa : pas) {
                String label = "";
                boolean firstArgAfterPredicate = true;
                int predicateIndex = pa.getPredicateIndex();

                //make sure predicate is a verb (conll08 data contains nominal predicates from NomaBank too)
                if (indexMap.int2str(posTags[predicateIndex]).startsWith("VB")) {

                    ArrayList<Argument> arguments = pa.getArguments();
                    boolean isAnyArgumentSeenAfterPredicate = isAnyArgumentSeenAfterPredicate(arguments);

                    if (isAnyArgumentSeenAfterPredicate == true) {
                        for (Argument ar : arguments) {

                            if (justCoreRoles == true) {
                                if (isACoreRole(ar.getType()) == true) {
                                    if (ar.getArgPosition() == ArgumentPosition.AFTER && firstArgAfterPredicate == true) {
                                        label += "p|" + ar.getType() + "|";
                                        firstArgAfterPredicate = false;
                                    } else
                                        label += ar.getType() + "|";
                                }

                            } else {
                                if (ar.getArgPosition() == ArgumentPosition.AFTER && firstArgAfterPredicate == true) {
                                    label += "p|" + ar.getType() + "|";
                                    firstArgAfterPredicate = false;
                                } else
                                    label += ar.getType() + "|";
                            }
                        }
                    } else {
                        for (Argument ar : arguments) {

                            if (justCoreRoles == true) {
                                if (isACoreRole(ar.getType()) == true) {
                                    label += ar.getType() + "|";
                                }

                            } else {
                                label += ar.getType() + "|";
                            }
                        }
                    }

                    if (!label.equals("")) {
                        if (isAnyArgumentSeenAfterPredicate == true)
                            argumentLabels.add(label.substring(0, label.length() - 1));
                        else
                            argumentLabels.add(label + "p");
                    }
                }

            }
        }

        return argumentLabels;
    }

    public static ArrayList<HashSet<String>> extractPredicateArgumentLabel_unorderedFormat(Sentence sen,
                                                                                           IndexMap indexMap,
                                                                                           boolean justMainPredicate,
                                                                                           boolean justCoreRoles) throws Exception {
        ArrayList<HashSet<String>> argumentLabels = new ArrayList<HashSet<String>>();
        ArrayList<PA> pas = sen.getPredicateArguments().getPredicateArgumentsAsArray();
        int[] posTags = sen.getPosTags();

        if (justMainPredicate == true) {
            boolean seenTheMainPredicate = false;
            for (PA pa : pas) {
                HashSet<String> label = new HashSet<String>();
                int predicateIndex = pa.getPredicateIndex();
                int predicateHeadIndex = sen.getDepHeads()[predicateIndex];

                if (predicateHeadIndex == 0) {
                    //make sure predicate is a verb (conll08 data contains nominal predicates from NomaBank too)

                    if (seenTheMainPredicate == true)
                        System.out.println("There are two predicates in this sentence with ROOT as their head!");

                    seenTheMainPredicate = true;

                    if (indexMap.int2str(posTags[predicateIndex]).startsWith("VB")) {
                        ArrayList<Argument> arguments = pa.getArguments();
                        for (Argument ar : arguments) {
                            if (justCoreRoles == true) {
                                if (isACoreRole(ar.getType()) == true) {
                                    label.add(ar.getType());
                                }
                            } else {
                                label.add(ar.getType());
                            }
                        }

                        if (label.size() > 0)
                            argumentLabels.add(label);
                    }
                }

            }

        } else {
            for (PA pa : pas) {
                HashSet<String> label = new HashSet<String>();
                int predicateIndex = pa.getPredicateIndex();

                //make sure predicate is a verb (conll08 data contains nominal predicates from NomaBank too)
                if (indexMap.int2str(posTags[predicateIndex]).startsWith("VB")) {

                    ArrayList<Argument> arguments = pa.getArguments();
                    for (Argument ar : arguments) {
                        if (justCoreRoles == true) {
                            if (isACoreRole(ar.getType()) == true) {
                                label.add(ar.getType());
                            }

                        } else {
                            label.add(ar.getType());
                        }
                    }

                    if (label.size() > 0)
                        argumentLabels.add(label);
                }

            }
        }

        return argumentLabels;
    }


    /*public static void updatePredArgLabelFreqDic(ArrayList<String> labels, Integer sentenceCounter){
        for (String label:labels)
        {
            if (predArgLabelFreqDic.containsKey(label))
            {
                ArrayList<Integer> tempArray= predArgLabelFreqDic.get(label);
                tempArray.set(0, predArgLabelFreqDic.get(label).get(0) + 1); //set the frequency
                tempArray.add(sentenceCounter);
                predArgLabelFreqDic.put(label, tempArray);
            }
            else
            {
                ArrayList<Integer> tempArray= new ArrayList<Integer>();
                tempArray.add(1);
                tempArray.add(sentenceCounter);
                predArgLabelFreqDic.put(label, tempArray);
            }
        }
    }*/


    public static void updatePredArgLabelFreqDic_unorderedFormat(ArrayList<HashSet<String>> labels, Integer sentenceCounter) {
        for (HashSet<String> label : labels) {
            if (duplicateKey(label)) {
                ArrayList<Integer> tempArray = predArgLabelFreqDic.get(label);
                tempArray.set(0, predArgLabelFreqDic.get(label).get(0) + 1); //set the frequency
                tempArray.add(sentenceCounter);
                predArgLabelFreqDic.put(label, tempArray);
            } else {
                ArrayList<Integer> tempArray = new ArrayList<Integer>();
                tempArray.add(1);
                tempArray.add(sentenceCounter);
                predArgLabelFreqDic.put(label, tempArray);
            }
        }
    }


   /* public static void printLabelFreq (String outputFile) throws IOException {
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile)));
        for (String label : predArgLabelFreqDic.keySet()) {
            int labelFreq = predArgLabelFreqDic.get(label).get(0);
            List<Integer> sentenceIDs = predArgLabelFreqDic.get(label).subList(1, predArgLabelFreqDic.get(label).size());

            writer.write(label + ":" + labelFreq + "\n");
            /*
            for (int i=0;i< sentenceIDs.size();i++)
            {
                int sentenceID= sentenceIDs.get(i);
                writer.write(sentenceID+":\n");

                //writing the sentence (for debugging purposes)
                String sentence= sentences.get(sentenceID);

                /*
                String[] tokens= sentence.split("\n");
                for (String token:tokens)
                {
                    String[] fields= token.split("\t");
                    String word= fields[1];
                    String predicate= fields[10];
                    String arguments="";

                    //extract arguments
                    if (fields.length>11) //we have at least one argument
                    {
                        for (int j=10;j<fields.length; j++){
                            arguments= fields[j]+"\t";
                        }
                    }
                    arguments= arguments.trim();

                    //write the sentence down
                    writer.write(word+"\t\t\t"+predicate+"\t\t\t"+ arguments+"\n");
                }

                writer.write("\n-----------\n");

                writer.write(sentence+"\n");
            }
            writer.write("\n");

        }
        writer.flush();
        writer.close();
    }
*/

    public static void printLabelFreq_unorderedFormat(String outputFile) throws IOException {
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile)));
        for (HashSet<String> label : predArgLabelFreqDic.keySet()) {
            int labelFreq = predArgLabelFreqDic.get(label).get(0);
            List<Integer> sentenceIDs = predArgLabelFreqDic.get(label).subList(1, predArgLabelFreqDic.get(label).size());

            writer.write(label + ":" + labelFreq + "\n");
            /*
            for (int i=0;i< sentenceIDs.size();i++)
            {
                int sentenceID= sentenceIDs.get(i);
                writer.write(sentenceID+":\n");

                //writing the sentence (for debugging purposes)
                String sentence= sentences.get(sentenceID);

                /*
                String[] tokens= sentence.split("\n");
                for (String token:tokens)
                {
                    String[] fields= token.split("\t");
                    String word= fields[1];
                    String predicate= fields[10];
                    String arguments="";

                    //extract arguments
                    if (fields.length>11) //we have at least one argument
                    {
                        for (int j=10;j<fields.length; j++){
                            arguments= fields[j]+"\t";
                        }
                    }
                    arguments= arguments.trim();

                    //write the sentence down
                    writer.write(word+"\t\t\t"+predicate+"\t\t\t"+ arguments+"\n");
                }

                writer.write("\n-----------\n");

                writer.write(sentence+"\n");


            }


            writer.write("\n");
            */
        }
        writer.flush();
        writer.close();
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


    public static boolean isAnyArgumentSeenAfterPredicate(ArrayList<Argument> args) {
        boolean isAnyArgumentSeenAfterPredicate = false;
        for (Argument arg : args) {
            if (arg.getArgPosition() == ArgumentPosition.BEFORE) {
                isAnyArgumentSeenAfterPredicate = true;
                break;
            }
        }
        return isAnyArgumentSeenAfterPredicate;
    }

    public static boolean duplicateKey(HashSet<String> newLabel) {
        for (HashSet<String> existingKey : predArgLabelFreqDic.keySet()) {
            if (isIdenticalHashSet(existingKey, newLabel) == true)
                return true;
        }
        return false;
    }

    public static boolean isIdenticalHashSet(HashSet<String> h1, HashSet<String> h2) {
        if (h1.size() != h2.size()) {
            return false;
        }
        HashSet<String> clone = new HashSet<String>(h2); // just use h2 if you don't need to save the original h2
        Iterator it = h1.iterator();

        while (it.hasNext()) {
            String temp = (String) it.next();
            if (clone.contains(temp)) { // replace clone with h2 if not concerned with saving data from h2
                clone.remove(temp);
            } else {
                return false;
            }
        }
        return true; // will only return true if sets are equal
    }


}
