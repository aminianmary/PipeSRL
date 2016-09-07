package SupervisedSRL.Strcutures;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by monadiab on 7/7/16.
 */
public class Prediction {
    String predicateLabel;
    HashMap<Integer, Integer> argumentLabels;

    public Prediction(String predicatedPredicateLabel, HashMap<Integer, Integer> predicatedArgumentLabels) {
        predicateLabel = predicatedPredicateLabel;
        argumentLabels = predicatedArgumentLabels;
    }

    public Prediction(String predicatedPredicateLabel, ArrayList<Integer> aiCandidIndices, ArrayList<Integer> acCandidLabels) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < aiCandidIndices.size(); i++) {
            int wordIdx = aiCandidIndices.get(i);
            int label = acCandidLabels.get(i);
            map.put(wordIdx, label);
        }
        predicateLabel = predicatedPredicateLabel;
        argumentLabels = map;
    }

    public String getPredicateLabel() {
        return predicateLabel;
    }

    public HashMap<Integer, Integer> getArgumentLabels() {
        return argumentLabels;
    }
}
