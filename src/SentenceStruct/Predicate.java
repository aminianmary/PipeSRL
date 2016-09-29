package SentenceStruct;

/**
 * Created by Maryam Aminian on 12/11/15.
 */
public class Predicate {

    int predicateIndex;
    String predicateLabel;


    public Predicate() {
        predicateIndex = -1;
        predicateLabel = "";
    }

    public Predicate(int givenIndex, String givenType) {
        predicateIndex = givenIndex;
        predicateLabel = givenType;
    }
    
    public int getIndex() {
        return predicateIndex;
    }

    public String getLabel() {
        return predicateLabel;
    }
}
