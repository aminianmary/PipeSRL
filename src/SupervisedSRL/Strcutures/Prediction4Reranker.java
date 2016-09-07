package SupervisedSRL.Strcutures;

import java.util.ArrayList;

/**
 * Created by monadiab on 8/24/16.
 */
public class Prediction4Reranker {
    String predicateLabel;
    ArrayList<Pair<Double, ArrayList<Integer>>> aiCandidates;
    ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandidates;

    public Prediction4Reranker(String predicateLabel, ArrayList<Pair<Double, ArrayList<Integer>>> aiCandids,
                               ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> acCandids) {
        this.predicateLabel = predicateLabel;
        this.aiCandidates = aiCandids;
        this.acCandidates = acCandids;
    }

    public String getPredicateLabel() {
        return predicateLabel;
    }

    public ArrayList<Pair<Double, ArrayList<Integer>>> getAiCandidates() {
        return aiCandidates;
    }

    public ArrayList<ArrayList<Pair<Double, ArrayList<Integer>>>> getAcCandidates() {
        return acCandidates;
    }

}
