package SentenceStruct;

import java.util.ArrayList;
import java.util.HashSet;

/**
 * Created by monadiab on 12/11/15.
 */
public class PA {

    Predicate pr;
    ArrayList<Argument> ams;

    public PA() {
        pr = new Predicate();
        ams = new ArrayList<Argument>();
    }

    public PA(Predicate p, ArrayList<Argument> a) {
        pr = p;
        ams = a;
    }

    public Predicate getPredicate() {
        return pr;
    }

    public void set(Predicate p) {
        pr = p;
    }

    public void updateArguments(Argument a) {
        ams.add(a);
    }

    public ArrayList<Argument> getArguments() {
        return ams;
    }

    public HashSet<Integer> getArgumentsIndices() {
        HashSet<Integer> argIndices = new HashSet<>();
        for (Argument a : ams)
            argIndices.add(a.getIndex());
        return argIndices;
    }

    public HashSet<PADependencyTuple> getAllPredArgDepTupls() {
        HashSet<PADependencyTuple> predArgDepTuples = new HashSet<PADependencyTuple>();
        int pIndex = pr.getIndex();
        for (Argument a : ams) {
            int aIndex = a.getIndex();
            String aType = a.getType();
            predArgDepTuples.add(new PADependencyTuple(pIndex, aIndex, aType));
        }

        return predArgDepTuples;
    }

    public String obtainArgumentType(int argIndex) {
        for (Argument arg : ams) {
            if (arg.getIndex() == argIndex)
                return arg.getType();
        }
        return "";
    }
}
