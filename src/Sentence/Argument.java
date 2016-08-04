package Sentence;

/**
 * Created by monadiab on 12/11/15.
 */
public class Argument {

    int index;
    String type;
    boolean seenBeforePredicate;

    public Argument() {
        index = -1;
        type = "";
        seenBeforePredicate = true;
    }

    public Argument(int givenIndex, String givenType, boolean isBeforePredicate) {
        index = givenIndex;
        type = givenType;
        seenBeforePredicate = isBeforePredicate;
    }

    public Argument(int givenIndex, String givenType) {
        index = givenIndex;
        type = givenType;
    }

    public boolean isSeenBeforePredicate() {
        return seenBeforePredicate;
    }

    public String getType() {
        return type;
    }

    public Integer getIndex() {
        return index;
    }

}
