package SupervisedSRL.Strcutures;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Set;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 2/5/15
 * Time: 10:27 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class CompactArray implements Serializable {
    HashMap<Integer, Double> array;

    public CompactArray(HashMap<Integer, Double> array) {
        this.array = array;
    }

    public CompactArray(int index, double initValue) {
        array = new HashMap<>();
        array.put(index, initValue);
    }

    public void expandArray(int index, double value) {
        if (array.containsKey(index))
            value += array.get(index);
        array.put(index, value);
    }

    public HashMap<Integer, Double> getArray() {
        return array;
    }

    public Set<Integer> keyset() {
        return array.keySet();
    }

    ;

    public double value(int i) {
        return array.containsKey(i) ? array.get(i) : 0;
    }
}