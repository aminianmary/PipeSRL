package SupervisedSRL.Reranker;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Created by Maryam Aminian on 8/24/16.
 */
public class RerankerInstanceItem implements Serializable {
    private HashMap<Integer, Integer>[] features;
    private String label;

    public RerankerInstanceItem(HashMap<Integer, Integer>[] features, String label) {
        this.features = features;
        this.label = label;
    }

    public HashMap<Integer, Integer>[] getFeatures() {
        return features;
    }

    public String getLabel() {
        return label;
    }
}
