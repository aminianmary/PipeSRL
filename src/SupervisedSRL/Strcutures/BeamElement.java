package SupervisedSRL.Strcutures;

/**
 * Created by Maryam Aminian on 5/24/16.
 */
public class BeamElement implements Comparable<BeamElement> {
    public int index;
    public double score;
    public int label;

    public BeamElement(int index, double score, int label) {
        this.index = index;
        this.score = score;
        this.label = label;
    }

    @Override
    public int compareTo(BeamElement beamElement) {
        double diff = score - beamElement.score;
        if (diff > 0)
            return 2;
        if (diff < 0)
            return -2;
        if (index != beamElement.index)
            return beamElement.index - index;
        return beamElement.label - label;
    }

    @Override
    public boolean equals(Object o) {
        return false;
    }
}