package SentenceStruct;

/**
 * Created by Maryam Aminian on 12/11/15.
 */
public class Predicate {

    int predicateIndex;
    String predicateGoldLabel = null;
    String predicateAutoLabel = null;


    public Predicate() {
        predicateIndex = -1;
        predicateGoldLabel = null;
        predicateAutoLabel = null;
    }

    public int getIndex() {
        return predicateIndex;
    }

    public String getPredicateAutoLabel() {
        return predicateAutoLabel;
    }

    public String getPredicateGoldLabel() {
        return predicateGoldLabel;
    }

    public void setPredicateIndex(int predicateIndex) {
        this.predicateIndex = predicateIndex;
    }

    public void setPredicateGoldLabel(String predicateGoldLabel) {
        this.predicateGoldLabel = predicateGoldLabel;
    }

    public void setPredicateAutoLabel(String predicateAutoLabel) {
        this.predicateAutoLabel = predicateAutoLabel;
    }
}
