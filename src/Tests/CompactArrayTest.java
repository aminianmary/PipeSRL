package Tests;

import SupervisedSRL.Strcutures.CompactArray;
import org.junit.Test;

/**
 * Created by monadiab on 8/3/16.
 */
public class CompactArrayTest {
    @Test
    public void testArrayExpansion() {
        CompactArray ca = new CompactArray(5, 4);
        ca.expandArray(2, 7);
        ca.expandArray(5, -3);
        assert ca.value(5) == 1;
    }
}
