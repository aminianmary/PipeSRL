package SupervisedSRL.Reranker;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by Maryam Aminian on 8/24/16.
 */
public class RerankerPool implements Serializable {
    private ArrayList<RerankerInstanceItem> items;
    private int goldIndex;

    public RerankerPool(ArrayList<RerankerInstanceItem> items, int goldIndex) {
        assert goldIndex < items.size() && goldIndex >= 0;
        this.items = items;
        this.goldIndex = goldIndex;
    }

    public RerankerPool() {
        items = new ArrayList<RerankerInstanceItem>();
    }

    public void addInstance(RerankerInstanceItem item, boolean isGold) {
        items.add(item);
        if (isGold) goldIndex = items.size() - 1;
    }

    public ArrayList<RerankerInstanceItem> getItems() {
        return items;
    }

    public int getGoldIndex() {
        return goldIndex;
    }

    public int length() {
        return items.size();
    }

    public RerankerInstanceItem item(int i) {
        return items.get(i);
    }

}
