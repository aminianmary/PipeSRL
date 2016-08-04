package Sentence;

import java.util.ArrayList;
import java.util.HashSet;

/**
 * Created by monadiab on 12/11/15.
 */
public class PAs {
    ArrayList<PA> predicateArguments;

    public PAs() {
        predicateArguments = new ArrayList<PA>();
    }

    public PAs(ArrayList<PA> pas) {
        predicateArguments = pas;
    }

    public void setPredicate(int predicateSeq, int predicateIndex, String predicateType) {
        if (predicateArguments.size() == predicateSeq) {
            //nothing about this predicate has seen before
            Predicate pr = new Predicate(predicateIndex, predicateType);
            PA pa = new PA(pr, new ArrayList<Argument>());
            predicateArguments.add(pa);
        } else if (predicateArguments.size() > predicateSeq) {
            //arguments of other predicates have been seen before
            PA currentPA = predicateArguments.get(predicateSeq);
            currentPA.set(new Predicate(predicateIndex, predicateType));
        } else if (predicateArguments.size() < predicateSeq) {
            //there is something wrong
            System.out.println("NOTE: " + predicateSeq + "th predicate is seen but previous predicates are not listed in the list!");
        }
    }


    public void setArgument(int associatedPredicateSeq, int argumentIndex, String argumentType) {
        if (predicateArguments.size() == associatedPredicateSeq) {
            //nothing about this predicate has seen before --> argument seen before predicate
            ArrayList<Argument> arguments = new ArrayList<Argument>();
            arguments.add(new Argument(argumentIndex, argumentType, true));
            PA pa = new PA(new Predicate(), arguments);
            predicateArguments.add(pa);
        } else if (predicateArguments.size() > associatedPredicateSeq) {
            //arguments/predicate of this predicates have been seen before
            //we still don't know has predicate seen before or not
            PA currentPA = predicateArguments.get(associatedPredicateSeq);

            if (currentPA.getPredicateIndex() > 0 && currentPA.getPredicateIndex() < argumentIndex)
                //check if we have seen predicate before this argument or not
                currentPA.updateArguments(new Argument(argumentIndex, argumentType, false));
            else
                currentPA.updateArguments(new Argument(argumentIndex, argumentType, true));
        } else if (predicateArguments.size() < associatedPredicateSeq) {
            //there is/are some arguments in between that are not observed in any ways (either their arguments or the predicate itself)
            for (int j = 0; j < (associatedPredicateSeq - predicateArguments.size() + 1); j++) {
                PA pa = new PA();
                predicateArguments.add(pa);
            }
            ArrayList<Argument> arguments = new ArrayList<Argument>();
            arguments.add(new Argument(argumentIndex, argumentType, true));
            PA pa = new PA(new Predicate(), arguments);
            predicateArguments.add(pa);
        }

    }


    public ArrayList<PA> getPredicateArgumentsAsArray() {
        return predicateArguments;
    }


    public HashSet<PADependencyTuple> getAllPredArgDepTupls() {
        ArrayList<PA> predArgs = this.getPredicateArgumentsAsArray();
        HashSet<PADependencyTuple> predArgDepTuples = new HashSet<PADependencyTuple>();
        for (PA pa : predArgs) {
            int pIndex = pa.getPredicateIndex();
            for (Argument a : pa.getArguments()) {
                int aIndex = a.getIndex();
                String aType = a.getType();

                predArgDepTuples.add(new PADependencyTuple(pIndex, aIndex, aType));
            }
        }

        return predArgDepTuples;
    }

}
