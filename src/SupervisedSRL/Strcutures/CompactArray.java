package SupervisedSRL.Strcutures;

import java.io.Serializable;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 2/5/15
 * Time: 10:27 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class CompactArray implements Serializable {
    double[] array;
    int offset;

    public CompactArray(int offset, double[] array) {
        this.offset = offset;
        this.array = array;
    }

    public void expandArray(int index, double value) {
        if (index < offset + array.length && index >= offset) {
            array[index - offset] += value;
        } else if (index < offset) {  //expand from left
            int gap = offset - index;
            int newSize = gap + array.length;
            double[] newArray = new double[newSize];
            newArray[0] = value;
            for (int i = 0; i < array.length; i++) {
                newArray[gap + i] = array[i];
            }
            this.offset = index;
            this.array = newArray;
        } else {
            int gap = index - (array.length + offset - 1);
            int newSize = array.length + gap;
            double[] newArray = new double[newSize];
            newArray[newSize - 1] = value;
            for (int i = 0; i < array.length; i++) {
                newArray[i] = array[i];
            }
            this.array = newArray;
        }
    }

    public double[] getArray() {
        return array;
    }

    public int getOffset() {
        return offset;
    }

    public int length() {
        return array.length;
    }
}