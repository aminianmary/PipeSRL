package SupervisedSRL.PD;

/**
 * Created by monadiab on 5/20/16.
 */
public class PredicateLexiconEntry implements Comparable {

    String plabel;
    Object[] pdfeats;

    public PredicateLexiconEntry(String givenLabel, Object[] computed_pdfeats) {
        plabel = givenLabel;
        pdfeats = computed_pdfeats;
    }


    public Object[] getPdfeats() {
        return pdfeats;
    }


    public String getPlabel() {
        return plabel;
    }


    @Override
    public boolean equals(Object obj) {
        if (obj instanceof PredicateLexiconEntry) {
            PredicateLexiconEntry ple = (PredicateLexiconEntry) obj;

            if (!ple.plabel.equals(plabel))
                return false;
            else if (!ple.getPdfeats().equals(pdfeats))
                return false;

            return true;
        }
        return false;
    }

    @Override
    public int compareTo(Object o) {
        if (equals(o))
            return 0;
        return hashCode() - o.hashCode();
    }

    @Override
    public int hashCode() {
        int hash = 0;
        hash ^= plabel.hashCode() * pdfeats.length;
        return hash;
    }


}
