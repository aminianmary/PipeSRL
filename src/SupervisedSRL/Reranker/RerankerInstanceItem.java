package SupervisedSRL.Reranker;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Created by Maryam Aminian on 8/24/16.
 */
public class RerankerInstanceItem implements Serializable {
    // todo this should be hashmap<int, int>
    private HashMap<Object, Integer>[] features;
    private String label;

    public RerankerInstanceItem(HashMap<Object, Integer>[] features, String label) {
        this.features = features;
        this.label = label;
    }

    public HashMap<Object, Integer>[] getFeatures() {
        return features;
    }

    public String getLabel() {
        return label;
    }
}
