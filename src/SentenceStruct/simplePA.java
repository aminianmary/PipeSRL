package SentenceStruct;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Maryam Aminian on 7/7/16.
 */
public class simplePA {
    String predicateLabel;
    HashMap<Integer, String> argumentLabels;

    public simplePA(String predicatedPredicateLabel, HashMap<Integer, String> predicatedArgumentLabels) {
        predicateLabel = predicatedPredicateLabel;
        argumentLabels = predicatedArgumentLabels;
    }

    public simplePA(String predicatedPredicateLabel, ArrayList<Integer> aiCandidIndices,
                    ArrayList<Integer> acCandidLabels, String[] labelMap) {
        HashMap<Integer, String> map = new HashMap<Integer, String>();
        for (int i = 0; i < aiCandidIndices.size(); i++) {
            int wordIdx = aiCandidIndices.get(i);
            int label = acCandidLabels.get(i);
            map.put(wordIdx, labelMap[label]);
        }
        predicateLabel = predicatedPredicateLabel;
        argumentLabels = map;
    }

    public String getPredicateLabel() {
        return predicateLabel;
    }

    public void setPredicateLabel(String pl) {
        predicateLabel = pl;
    }

    public HashMap<Integer, String> getArgumentLabels() {
        return argumentLabels;
    }
}
