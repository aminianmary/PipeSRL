package Sentence;

/**
 * Created by monadiab on 12/11/15.
 */
public class Argument {
    int index;
    String type;
    ArgumentPosition argPosition;

    public Argument() {
        index = -1;
        type = "";
        argPosition = ArgumentPosition.BEFORE;
    }

    public Argument(int givenIndex, String givenType, ArgumentPosition position) {
        index = givenIndex;
        type = givenType;
        argPosition = position;
    }

    public Argument(int givenIndex, String givenType) {
        index = givenIndex;
        type = givenType;
    }

    public ArgumentPosition getArgPosition() {
        return argPosition;
    }

    public String getType() {
        return type;
    }

    public Integer getIndex() {
        return index;
    }

}
