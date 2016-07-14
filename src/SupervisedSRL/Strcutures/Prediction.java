package SupervisedSRL.Strcutures;

import com.sun.jdi.InvalidTypeException;

import java.util.HashMap;

/**
 * Created by monadiab on 7/7/16.
 */
public class Prediction {
    String predicateLabel;
    HashMap<Integer, Integer> argumentLabels;

    public Prediction()
    {
        predicateLabel= "";
        argumentLabels= new HashMap<Integer, Integer>();
    }
    public Prediction(String predicatedPredicateLabel, HashMap<Integer, Integer> predicatedArgumentLabels)
    {
        predicateLabel= predicatedPredicateLabel;
        argumentLabels = predicatedArgumentLabels;
    }

    public String getPredicateLabel() {return predicateLabel;}

    public HashMap<Integer, Integer> getArgumentLabels() {return argumentLabels;}
}
