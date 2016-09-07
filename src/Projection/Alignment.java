package Projection;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

/**
 * Created by monadiab on 3/23/16.
 */
public class Alignment {

    private HashMap<Integer, HashMap<Integer, Integer>> sourceTargetAlignmentDic;
    private HashMap<Integer, HashMap<Integer, Integer>> targetSourceAlignmentDic;


    public Alignment(String alignmentFilePath) throws IOException {
        Object[] obj = createAlignmentDic(alignmentFilePath);
        sourceTargetAlignmentDic = (HashMap<Integer, HashMap<Integer, Integer>>) obj[0];
        targetSourceAlignmentDic = (HashMap<Integer, HashMap<Integer, Integer>>) obj[1];
    }

    public HashMap<Integer, HashMap<Integer, Integer>> getSourceTargetAlignmentDic() {
        return sourceTargetAlignmentDic;
    }

    public HashMap<Integer, HashMap<Integer, Integer>> getTargetSourceAlignmentDic() {
        return targetSourceAlignmentDic;
    }

    public Object[] createAlignmentDic
            (String alignmentFile) throws IOException {

        BufferedReader alignmentReader = new BufferedReader(new FileReader(alignmentFile));
        HashMap<Integer, HashMap<Integer, Integer>> alignmentDic = new HashMap<Integer, HashMap<Integer, Integer>>();
        HashMap<Integer, HashMap<Integer, Integer>> alignmentDicReverse = new HashMap<Integer, HashMap<Integer, Integer>>();


        String alignLine2Read = "";

        int sentenceID = -1;

        while (((alignLine2Read = alignmentReader.readLine()) != null)) {
            sentenceID++;

            //logging
            if (sentenceID % 10000 == 0)
                System.out.print(sentenceID);
            else if (sentenceID % 1000 == 0)
                System.out.print(".");

            alignmentDic.put(sentenceID, new HashMap<Integer, Integer>());
            alignmentDicReverse.put(sentenceID, new HashMap<Integer, Integer>());


            if (!alignLine2Read.trim().equals("")) {
                String[] alignWords = alignLine2Read.split(" ");

                for (int i = 0; i < alignWords.length; i++) {
                    // finding indexes from alignment data
                    Integer sourceIndex = Integer.parseInt(alignWords[i].toString().split("-")[0]);
                    Integer targetIndex = Integer.parseInt(alignWords[i].split("-")[1]);

                    if (!alignmentDic.get(sentenceID).containsKey(sourceIndex))
                        alignmentDic.get(sentenceID).put(sourceIndex, targetIndex);
                    else {
                        //removes the noisy alignment
                        alignmentDic.get(sentenceID).remove(sourceIndex);
                        System.out.println("SentenceStruct: " + sentenceID + " source index: " + sourceIndex + " is aligned to multiple target words");
                    }

                    if (!alignmentDicReverse.get(sentenceID).containsKey(targetIndex))
                        alignmentDicReverse.get(sentenceID).put(targetIndex, sourceIndex);
                }

            } else {
                if (alignLine2Read.equals("")) {
                    System.out.println("SentenceStruct " + sentenceID + ": alignment is empty");
                }
            }
        }

        alignmentReader.close();
        return new Object[]{alignmentDic, alignmentDicReverse};
    }
}
