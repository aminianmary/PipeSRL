package util;

import Sentence.Sentence;
import SupervisedSRL.Strcutures.IndexMap;
import SupervisedSRL.Strcutures.Prediction;
import apple.laf.JRSUIUtils;

import java.io.*;
import java.util.*;

/**
 * Created by monadiab on 4/12/16.
 */
public class IO {

    public static ArrayList<String> readCoNLLFile(String coNLLFile) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(coNLLFile)));
        ArrayList<String> sentences = new ArrayList<String>();
        String line2read = "";
        int counter=0;
        StringBuilder sentence = new StringBuilder();
        while ((line2read = reader.readLine()) != null) {
            if (line2read.equals("")) //sentence break
            {
                counter++;

                if (counter%100000==0)
                    System.out.print(counter);
                else if (counter%10000==0)
                    System.out.print(".");

                String senText = sentence.toString().trim();
                if(senText.length()>0)
                    sentences.add(senText);
                sentence = new StringBuilder();
            } else {
                sentence.append(line2read);
                sentence.append("\n");
            }
        }
        return sentences;
    }

    public static ArrayList<String> readCoNLLFile_into_words(String coNLLFile) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(coNLLFile)));
        ArrayList<String> sentences = new ArrayList<String>();
        String line2read = "";
        String sentence = "";

        while ((line2read = reader.readLine()) != null) {
            if (line2read.equals("")) //sentence break
            {
                sentences.add(sentence.trim());
                sentence = "";
            } else {
                sentence += line2read.split("\t")[1] + " ";
            }
        }
        return sentences;
    }


    public static ArrayList<String> readPlainFile(String plainFile) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(plainFile)));
        ArrayList<String> sentences = new ArrayList<String>();
        String line2read = "";

        while ((line2read = reader.readLine()) != null) {

            sentences.add(line2read.trim());
        }
        return sentences;
    }


    public static Object[] readCoNLLFile_into_words_and_conll_data (String coNLLFile ) throws IOException
    {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(coNLLFile)));
        ArrayList<String> sentences_conll = new ArrayList<String>();
        ArrayList<String> sentences_words = new ArrayList<String>();

        String line2read = "";
        String sentence_conll = "";
        String sentence_words = "";

        int sentenceCounter=0;
        while ((line2read = reader.readLine()) != null) {
            if (line2read.equals("")) //sentence break
            {
                sentenceCounter++;

                if(sentenceCounter %100000==0)
                    System.out.print(sentenceCounter);
                else if (sentenceCounter%10000==0)
                    System.out.print(".");

                sentences_conll.add(sentence_conll);
                sentences_words.add(sentence_words.trim());
                sentence_conll = "";
                sentence_words= "";
            } else {
                sentence_conll += line2read + "\n";
                sentence_words += line2read.split("\t")[1]+" ";
            }
        }

        return new Object[]{sentences_conll, sentences_words};
    }


    public static void writePredictionsInCoNLLFormat (ArrayList<ArrayList<String>> sentencesForOutput,
                                                     TreeMap<Integer, Prediction>[] predictedPAs,
                                                      String[] labelMap,
                                                      String outputFile)
            throws IOException
    {
        BufferedWriter outputWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile)));
        for (int d=0; d< sentencesForOutput.size(); d++)
        {
            ArrayList<String> sentenceForOutput = sentencesForOutput.get(d);
            TreeMap<Integer, Prediction> predictionForThisSentence = predictedPAs[d];

            for (int wordIdx=0; wordIdx< sentenceForOutput.size(); wordIdx++)
            {
                int realWordIdx = wordIdx+1;
                //for each word in the sentence
                outputWriter.write(sentenceForOutput.get(wordIdx)+"\t");  //filling fields 0-11

                if (predictionForThisSentence.containsKey(realWordIdx)) {
                    Prediction prediction = predictionForThisSentence.get(realWordIdx);
                    //this is a predicate
                    outputWriter.write("Y\t"); //filed 12
                    outputWriter.write(prediction.getPredicateLabel()); //field 13
                }else
                {
                    //this is not a predicate
                    outputWriter.write("_\t"); //filed 12
                    outputWriter.write("_"); //field 13
                }

                //checking if this word has been an argument for other predicates or not (fields 14-end)
                for (int pIdx:predictionForThisSentence.keySet())
                {
                    HashMap<Integer, Integer> argumentLabels = predictionForThisSentence.get(pIdx).getArgumentLabels();
                    if (argumentLabels.containsKey(realWordIdx)) {
                        if (!labelMap[argumentLabels.get(realWordIdx)].equals("0"))
                            //word is an argument
                            outputWriter.write("\t" + labelMap[argumentLabels.get(realWordIdx)]);
                        else
                            //word is not an argument for this predicate
                            outputWriter.write("\t_");
                    }
                    else
                        //word is not an argument for this predicate
                        outputWriter.write("\t_");
                }

                outputWriter.write("\n");
            }
            outputWriter.write("\n");
        }
        outputWriter.flush();
        outputWriter.close();
    }


    public static String formatString2Conll (String input)
    {
        String ConllFormatString="";
        int wordIdx=0;
        for (String word: input.split(" ")) {
            wordIdx++;
            ConllFormatString += wordIdx+"\t"+ word+"\n";
        }

        return ConllFormatString;
    }


    public static ArrayList<String> getSentenceForOutput (String sentenceInCoNLLFormat)
    {
        String[] lines = sentenceInCoNLLFormat.split("\n");
        ArrayList<String> sentenceForOutput = new ArrayList<String>();
        for (String line:lines)
        {
            String[] fields = line.split("\t");
            String filedsForOutput = "";
            //we just need the first 12 fields. The rest of filed must be filled based on what system predicted
            for (int k=0; k< 12; k++)
                filedsForOutput += fields[k]+"\t";
            sentenceForOutput.add(filedsForOutput.trim());
        }
        return sentenceForOutput;
    }


    public static HashSet<String> obtainLabels (List<String> sentences) {
        System.out.println("Getting set of labels...");
        HashSet<String> labels = new HashSet<String>();

        int counter = 0;
        for (String sentence : sentences) {
            counter++;
            if (counter % 1000 == 0)
                System.out.println(counter + "/" + sentences.size());

            String[] tokens = sentence.trim().split("\n");
            for (String token : tokens) {
                String[] fields= token.split("\t");
                for (int k=14 ; k< fields.length; k++)
                    if (!fields[k].equals("_"))
                        labels.add(fields[k]);
            }
        }
        return labels;
    }

}