package Tests;

import SupervisedSRL.Strcutures.CompactArray;
import org.junit.Test;

/**
 * Created by monadiab on 8/3/16.
 */
public class CompactArrayTest {
    @Test
    public void testArrayExpansion() {
        CompactArray ca = new CompactArray(5, new double[]{4, 0, 2});
        ca.expandArray(2, 7);
        assert equals(ca.getArray(), new double[]{7, 0, 0, 4, 0, 2});
        ca.expandArray(3, 5);
        assert equals(ca.getArray(), new double[]{7, 5, 0, 4, 0, 2});
    }

    private boolean equals(double[] a1, double[] a2) {
        if (a1.length != a2.length) return false;
        for (int i = 0; i < a1.length; i++)
            if (a1[i] != a2[i])
                return false;
        return true;
    }
}
