package SupervisedSRL.Strcutures;

import com.sun.jdi.InvalidTypeException;

import java.util.HashMap;

/**
 * Created by monadiab on 7/7/16.
 */
public class Prediction {
    String predicateLabel;
    HashMap<Integer, String> argumentLabels;

    public Prediction()
    {
        predicateLabel= "";
        argumentLabels= new HashMap<Integer, String>();
    }
    public Prediction(String predicatedPredicateLabel, HashMap<Integer, String> predicatedArgumentLabels)
    {
        predicateLabel= predicatedPredicateLabel;
        argumentLabels = predicatedArgumentLabels;
    }

    public String getPredicateLabel() {return predicateLabel;}

    public HashMap<Integer, String> getArgumentLabels() {return argumentLabels;}
}
