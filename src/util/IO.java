package util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

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
}